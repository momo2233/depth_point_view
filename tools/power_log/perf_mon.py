#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""perf_mon.py — 终端实时性能监测（CPU / GPU 计算 / 内存 / 显存 / 温度 / 功耗）

运行后在当前窗口原地刷新，Ctrl+C 结束并打印这段时间的 平均 / 峰值 / 最低 汇总。

特点
  * 纯标准库，无第三方依赖（python3 即可跑）
  * 自动选择 GPU 后端：
      - x86 + 独显   : nvidia-smi（GPU 计算占用率、独立显存、温度、功耗、SM 频率）
      - Jetson       : tegrastats（GR3D_FREQ、统一内存、温度、VDD_IN 功耗）
      - 兜底         : sysfs devfreq load
  * -p 可顺带统计某进程（PID 或进程名）的 CPU% 与 RSS，进程重启会自动重新捕获
  * --csv 可同时落一份采样表，便于事后画图
  * Jetson 的「显存」取 nvmap 的 GPU 侧真实分配（同 jtop 进程表的 [GPU MEM] 合计），
    该文件在 debugfs 只有 root 可读，所以板上要用 sudo 运行才看得到；普通用户会显示提示
  * 功耗与 jtop 的 ALL 同口径：有 VDD_IN 用 VDD_IN，否则所有电源轨累加

用法
  ./perf_mon.py                     # 一直跑，Ctrl+C 结束
  ./perf_mon.py -i 0.5              # 0.5s 采样
  ./perf_mon.py -d 60               # 跑 60 秒自动结束（便于挂在脚本里）
  ./perf_mon.py -p camcap_capture   # 顺带盯采集节点的 CPU/内存
  ./perf_mon.py -p 11341 --csv run.csv
"""

import argparse
import csv
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- 终端样式
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {"g": "\033[32m", "y": "\033[33m", "r": "\033[31m", "c": "\033[36m", "m": "\033[35m"}
USE_COLOR = True


def col(text, code):
    if not USE_COLOR or code not in COLORS:
        return str(text)
    return COLORS[code] + str(text) + RESET


def level(pct):
    """按占用率给颜色：<50 绿，<80 黄，>=80 红。"""
    if pct is None:
        return "c"
    if pct >= 80:
        return "r"
    if pct >= 50:
        return "y"
    return "g"


def bar(pct, width=22):
    pct = 0.0 if pct is None else max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "░" * (width - filled)


def gib(nbytes):
    return 0.0 if nbytes is None else nbytes / 1024.0 ** 3


def hms(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def dwidth(text):
    """显示宽度：中日韩全角字符按 2 列算，用于对齐。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text, width):
    return text + " " * max(0, width - dwidth(text))


# ---------------------------------------------------------------- CPU
class CpuSampler:
    def __init__(self):
        self.prev_total, self.prev_cores = self._read()

    @staticmethod
    def _read():
        total, cores = None, []
        try:
            with open("/proc/stat") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "cpu":
                        total = [int(x) for x in parts[1:]]
                    elif parts[0].startswith("cpu") and parts[0][3:].isdigit():
                        cores.append([int(x) for x in parts[1:]])
                    else:
                        break
        except OSError:
            pass
        return total or [0] * 8, cores

    @staticmethod
    def _usage(prev, cur):
        idle_prev = prev[3] + (prev[4] if len(prev) > 4 else 0)
        idle_cur = cur[3] + (cur[4] if len(cur) > 4 else 0)
        dt = sum(cur) - sum(prev)
        if dt <= 0:
            return 0.0
        busy = dt - (idle_cur - idle_prev)
        return max(0.0, min(100.0, 100.0 * busy / dt))

    def sample(self):
        cur_total, cur_cores = self._read()
        overall = self._usage(self.prev_total, cur_total)
        hot = 0.0
        for i in range(min(len(self.prev_cores), len(cur_cores))):
            hot = max(hot, self._usage(self.prev_cores[i], cur_cores[i]))
        self.prev_total, self.prev_cores = cur_total, cur_cores
        return {"cpu": overall, "cpu_hot": hot, "cores": len(cur_cores)}


# ---------------------------------------------------------------- 内存
def read_meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                try:
                    info[key] = int(rest.split()[0]) * 1024
                except (IndexError, ValueError):
                    continue
    except OSError:
        pass
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    return {
        "mem_total": total,
        "mem_used": max(0, total - avail),
        "swap_total": swap_total,
        "swap_used": max(0, swap_total - swap_free),
    }


# ---------------------------------------------------------------- GPU 显存（nvmap）
# Jetson 上 GPU 没有独立显存，但 nvmap 记录了「GPU 侧真实分配了多少」，
# 也就是 jtop 进程表里 [GPU MEM] 那一列。文件在 debugfs，只有 root 可读。
NV_MAPS = "/sys/kernel/debug/nvmap/iovmm/maps"
_NV_LINE = re.compile(r"^(?P<user>\w+)\s+(?P<process>\S+)\s+(?P<pid>\d+)\s+(?P<size>\d+)(?P<unit>\w)")
_NV_TOTAL = re.compile(r"^total\s+(?P<size>\d+)(?P<unit>\w)")


def read_nvmap():
    """返回 {"gpu_mem_total": bytes, "gpu_mem_by_pid": {pid: bytes}}；读不到返回 None。"""
    try:
        fp = open(NV_MAPS, "r")
    except OSError:
        return None
    total = None
    by_pid = {}
    with fp:
        for line in fp:
            m = _NV_LINE.search(line)
            if m:
                pid = int(m.group("pid"))
                by_pid[pid] = by_pid.get(pid, 0) + int(m.group("size")) * 1024
                continue
            m = _NV_TOTAL.search(line)
            if m:
                total = int(m.group("size")) * 1024
    if total is None:
        total = sum(by_pid.values())
    return {"gpu_mem_total": total, "gpu_mem_by_pid": by_pid}


# ---------------------------------------------------------------- GPU 后端
class NvidiaSmiGpu:
    """x86 + 独显。"""

    name = "nvidia-smi"

    def __init__(self):
        self.path = shutil.which("nvidia-smi")

    @staticmethod
    def _num(txt):
        try:
            return float(str(txt).strip())
        except (TypeError, ValueError):
            return None

    def available(self):
        """Jetson 上同样有 nvidia-smi，但查询返回 [N/A]，必须实测一次才能决定用不用。"""
        if not self.path:
            return False
        try:
            out = subprocess.run(
                [self.path, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return False
        rows = out.splitlines()
        return bool(rows) and self._num(rows[0]) is not None

    def start(self):
        pass

    def close(self):
        pass

    def sample(self):
        query = ("utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw,clocks.current.sm")
        try:
            out = subprocess.run(
                [self.path, "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        rows = out.strip().splitlines()
        if not rows:
            return {}
        vals = [self._num(v) for v in rows[0].split(",")]
        vals += [None] * (6 - len(vals))
        d = {}
        if vals[0] is not None:
            d["gpu"] = vals[0]
        if vals[1] is not None:
            d["vram_used"] = vals[1] * 1024 * 1024
        if vals[2] is not None:
            d["vram_total"] = vals[2] * 1024 * 1024
        if vals[3] is not None:
            d["gpu_temp"] = vals[3]
        if vals[4] is not None:
            d["power"] = vals[4]
        if vals[5] is not None:
            d["gpu_clk"] = vals[5]
        return d


class TegrastatsGpu:
    """Jetson：tegrastats 流式解析（GR3D_FREQ / RAM / 温度 / VDD_IN）。"""

    name = "tegrastats"

    def __init__(self, interval_ms=1000):
        self.interval_ms = max(50, int(interval_ms))
        self.proc = None
        self.lock = threading.Lock()
        self.latest = {}

    @staticmethod
    def available():
        return os.path.exists("/etc/nv_tegra_release") and shutil.which("tegrastats") is not None

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except OSError:
            self.proc = None
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self._parse(line)
        except (OSError, ValueError):
            pass

    def _parse(self, line):
        d = {}
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            d["ram_used"] = int(m.group(1)) * 1024 * 1024
            d["ram_total"] = int(m.group(2)) * 1024 * 1024
        m = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            d["gpu"] = float(m.group(1))
        # 温度标签各版本大小写不一（GPU@50C / gpu@66.5C），统一按忽略大小写匹配
        m = re.search(r"\bgpu@([\d.]+)C", line, re.I)
        if m:
            d["gpu_temp"] = float(m.group(1))
        m = re.search(r"\bcpu@([\d.]+)C", line, re.I)
        if m:
            d["cpu_temp"] = float(m.group(1))
        m = re.search(r"\btj@([\d.]+)C", line, re.I)
        if m:
            d["soc_temp"] = float(m.group(1))
        # 功耗：与 jtop 的 ALL 同口径 —— 有 VDD_IN 时它就是整板总输入，直接用；
        # 否则把所有电源轨累加（Orin 开发套件：VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0）
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            d["power"] = int(m.group(1)) / 1000.0
            d["power_rails"] = {"IN": d["power"]}
        else:
            rails = {}
            for rail, val in re.findall(r"\b((?:VDD|VIN)_[A-Z0-9_]+)\s+(\d+)mW", line):
                name = rail.replace("VDD_", "").replace("VIN_", "")
                rails[name] = int(val) / 1000.0
            if rails:
                d["power_rails"] = rails
                d["power"] = round(sum(rails.values()), 2)
        m = re.search(r"EMC_FREQ (\d+)%", line)
        if m:
            d["emc"] = float(m.group(1))
        if d:
            d["_ts"] = time.time()
            with self.lock:
                self.latest.update(d)

    def sample(self):
        with self.lock:
            return {k: v for k, v in self.latest.items() if k != "_ts"}

    def wait_ready(self, timeout=3.0):
        """等第一行 tegrastats 数据到位，避免首帧还没有 GPU/温度/功耗。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.latest:
                    return True
            time.sleep(0.05)
        return False

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except (OSError, subprocess.SubprocessError):
            try:
                self.proc.kill()
            except OSError:
                pass


class SysfsGpu:
    """兜底：devfreq load（0..1000 换算成百分比）。"""

    name = "sysfs"
    LOAD = ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
            "/sys/class/devfreq/gpu/load"]

    @staticmethod
    def available():
        return any(os.path.exists(p) for p in SysfsGpu.LOAD)

    def start(self):
        pass

    def close(self):
        pass

    def sample(self):
        for path in self.LOAD:
            try:
                with open(path) as f:
                    return {"gpu": min(100.0, float(f.read().strip()) / 10.0)}
            except (OSError, ValueError):
                continue
        return {}


# ---------------------------------------------------------------- 单进程追踪
class ProcSampler:
    def __init__(self, spec):
        self.spec = str(spec)
        self.pid = self._resolve(self.spec)
        self.clk = os.sysconf("SC_CLK_TCK")
        self.prev = None
        self.last = time.time()

    @staticmethod
    def _resolve(spec):
        if spec.isdigit():
            return int(spec)
        # 先按进程名精确匹配；再退回 -f 全命令行匹配。
        # 两种方式都要排除自己和父进程，否则 `-p 某进程` 这段命令行本身会被 pgrep -f 命中。
        for pgrep_args in (["pgrep", "-x", spec], ["pgrep", "-f", spec]):
            try:
                out = subprocess.run(pgrep_args, capture_output=True,
                                     text=True, timeout=3).stdout.split()
            except (OSError, subprocess.SubprocessError):
                out = []
            for token in out:
                try:
                    pid = int(token)
                except ValueError:
                    continue
                if pid in (os.getpid(), os.getppid()):
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
                except OSError:
                    continue
                if "perf_mon" in cmdline:
                    continue
                return pid
        return None

    def _times(self):
        with open(f"/proc/{self.pid}/stat") as f:
            raw = f.read()
        rest = raw.rsplit(")", 1)[1].split()
        return int(rest[11]), int(rest[12])  # utime, stime

    def _name(self):
        try:
            with open(f"/proc/{self.pid}/comm") as f:
                return f.read().strip()
        except OSError:
            return "?"

    def sample(self):
        now = time.time()
        if self.pid is None:
            if not self.spec.isdigit():
                self.pid = self._resolve(self.spec)
            if self.pid is None:
                return {"proc_name": self.spec, "proc_pid": None, "proc_cpu": 0.0,
                        "proc_rss": 0, "proc_dead": True}
        try:
            utime, stime = self._times()
        except (OSError, IndexError, ValueError):
            self.prev = None
            if not self.spec.isdigit():
                self.pid = self._resolve(self.spec)
            return {"proc_name": self.spec, "proc_pid": self.pid, "proc_cpu": 0.0,
                    "proc_rss": 0, "proc_dead": True}
        dt = now - self.last
        cpu = 0.0
        if self.prev is not None and dt > 0:
            cpu = 100.0 * ((utime + stime) - (self.prev[0] + self.prev[1])) / self.clk / dt
        self.prev = (utime, stime)
        self.last = now
        rss = 0
        try:
            with open(f"/proc/{self.pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        return {"proc_name": self._name(), "proc_pid": self.pid,
                "proc_cpu": max(0.0, cpu), "proc_rss": rss, "proc_dead": False}


# ---------------------------------------------------------------- 汇总统计
STAT_KEYS = ("cpu", "cpu_sum", "cpu_hot", "gpu", "mem_used", "vram_used",
             "gpu_temp", "cpu_temp", "soc_temp", "power", "proc_cpu", "proc_rss",
             "proc_gpu_mem")


class Stats:
    def __init__(self):
        self.n = 0
        self.last = {}
        self.acc = {k: [] for k in STAT_KEYS}

    def add(self, s):
        self.n += 1
        self.last = s
        for k in STAT_KEYS:
            v = s.get(k)
            if isinstance(v, (int, float)):
                self.acc[k].append(float(v))

    def get(self, k):
        vals = self.acc.get(k) or []
        if not vals:
            return None
        return {"avg": sum(vals) / len(vals), "max": max(vals), "min": min(vals)}


# ---------------------------------------------------------------- 屏幕
class Screen:
    """原地刷新：每次把上一次写出的行数移回去重写。"""

    def __init__(self):
        self.prev = 0

    def draw(self, lines):
        n = max(len(lines), self.prev)
        lines = list(lines) + [""] * (n - len(lines))
        buf = ["\033[%dA" % self.prev] if self.prev else []
        for line in lines:
            buf.append("\033[2K" + line + "\n")
        sys.stdout.write("".join(buf))
        sys.stdout.flush()
        self.prev = n


# ---------------------------------------------------------------- 监视器
class Monitor:
    def __init__(self, interval, pid_spec=None):
        self.interval = interval
        self.cpu = CpuSampler()
        self.gpu = self._pick_gpu(interval)
        self.unified = isinstance(self.gpu, TegrastatsGpu)
        if self.gpu is not None:
            self.gpu.start()
            wait = getattr(self.gpu, "wait_ready", None)
            if wait is not None:
                wait(max(2.0, interval * 2))
        self.proc = ProcSampler(pid_spec) if pid_spec else None

    @staticmethod
    def _pick_gpu(interval):
        jetson = os.path.exists("/etc/nv_tegra_release")
        teg = TegrastatsGpu(interval * 1000)
        smi = NvidiaSmiGpu()
        order = [teg, smi] if jetson else [smi, teg]
        for backend in order:
            if backend.available():
                return backend
        sysg = SysfsGpu()
        if sysg.available():
            return sysg
        return None

    def sample(self):
        s = {"t": time.time()}
        s.update(self.cpu.sample())
        # cpu 是整机占比（0..100）；cpu_sum 是各核累加（1 核满载 = 100，N 核 = N*100）
        cores = s.get("cores") or 1
        s["cpu_sum"] = s.get("cpu", 0.0) * cores
        s.update(read_meminfo())
        if self.gpu is not None:
            s.update(self.gpu.sample())
        # 独立显存（nvidia-smi）优先；否则用 nvmap 的 GPU 侧真实分配（同 jtop [GPU MEM]）
        if s.get("vram_used") is None:
            nv = read_nvmap()
            if nv is not None:
                s.update(nv)
                s["vram_used"] = nv["gpu_mem_total"]
                s["vram_total"] = s.get("mem_total")
                s["vram_kind"] = "nvmap"
        if self.proc is not None:
            s.update(self.proc.sample())
            if s.get("proc_pid") is not None:
                s["proc_gpu_mem"] = (s.get("gpu_mem_by_pid") or {}).get(s["proc_pid"], 0)
        try:
            s["load1"] = os.getloadavg()[0]
        except OSError:
            pass
        return s

    def close(self):
        if self.gpu is not None:
            self.gpu.close()


# ---------------------------------------------------------------- 渲染
LW, NW, BW = 13, 21, 20   # 标签列宽 / 数值列宽 / 进度条宽度


def _row(label, num_text, pct, tail="", show_bar=True):
    """标签左对齐 + 数值右对齐 + 彩色进度条，保证各行进度条起始列一致。"""
    out = pad(label, LW) + col(num_text.rjust(NW), level(pct))
    if show_bar:
        out += "  " + col(bar(pct, BW), level(pct))
    if tail:
        out += "  " + tail
    return out


def render(s, st, elapsed, interval, backend, host):
    lines = [col("─" * 84, "c")]
    lines.append(f"{col('perf_mon', 'c')} · {host} · {backend} · "
                 f"间隔 {interval:.1f}s · 已运行 {hms(elapsed)} · 采样 {st.n}")

    cpu = s.get("cpu", 0.0)
    cores = s.get("cores") or 1
    # 主指标：各核累加（1 核满载 = 100%，N 核满载 = N*100%），进度条按满核归一
    lines.append(_row("CPU 总占用", "%.1f%%" % s.get("cpu_sum", cpu * cores), cpu,
                      "满核 %d%%（%d 核累加）" % (cores * 100, cores)))
    lines.append(_row("CPU 整机占用", "%.1f%%" % cpu, cpu,
                      "%d 核 · 最忙核 %5.1f%% · load %.2f" % (
                          cores, s.get("cpu_hot", 0.0), s.get("load1", 0.0))))

    gpu = s.get("gpu")
    if gpu is None:
        lines.append(_row("GPU 计算", "--", None, col("未检测到 GPU 占用率接口", "r")))
    else:
        extra = []
        if s.get("gpu_clk") is not None:
            extra.append("%d MHz" % s["gpu_clk"])
        if s.get("emc") is not None:
            extra.append("EMC %d%%" % s["emc"])
        lines.append(_row("GPU 计算", "%5.1f%%" % gpu, gpu, "  ".join(extra)))

    mt, mu = s.get("mem_total", 0), s.get("mem_used", 0)
    mp = 100.0 * mu / mt if mt else 0.0
    swap = ""
    if s.get("swap_total"):
        swap = "swap %.1f/%.1f GiB" % (gib(s.get("swap_used", 0)), gib(s["swap_total"]))
    lines.append(_row("内存", "%.1f/%.1f GiB  %.1f%%" % (gib(mu), gib(mt), mp), mp, swap))

    vt, vu = s.get("vram_total"), s.get("vram_used")
    if vu is None:
        hint = "需 root 读 nvmap（用 sudo 运行）" if backend == "tegrastats" else "平台无 GPU 显存统计"
        lines.append(_row("显存", "--", None, col(hint, "r"), show_bar=False))
    elif s.get("vram_kind") == "nvmap":
        vp = 100.0 * vu / vt if vt else 0.0
        lines.append(_row("显存", "%.1f MiB  %.2f%%" % (vu / 1024.0 ** 2, vp), vp,
                          col("nvmap 的 GPU 侧分配（同 jtop [GPU MEM] 合计）", "m"),
                          show_bar=False))
    else:
        vp = 100.0 * vu / vt if vt else 0.0
        lines.append(_row("显存", "%.1f/%.1f GiB  %.1f%%" % (gib(vu), gib(vt), vp), vp,
                          col("总 %.1f GiB" % gib(vt), "m")))

    temps = []
    for key, label in (("cpu_temp", "CPU"), ("gpu_temp", "GPU"), ("soc_temp", "SoC")):
        if s.get(key) is not None:
            temps.append("%s %.1f°C" % (label, s[key]))
    pw = ""
    if s.get("power") is not None:
        pw = "功耗 %.1f W" % s["power"]
        rails = s.get("power_rails") or {}
        if len(rails) > 1:
            pw += "（%s）" % " + ".join("%s %.1f" % (k, v) for k, v in rails.items())
    if temps or pw:
        lines.append(pad("温度", LW) + col("  ".join(temps), "c") + "    " + col(pw, "m"))

    if "proc_pid" in s:
        if s.get("proc_dead"):
            lines.append(pad("进程", LW) + "%s  %s" % (s.get("proc_name", "?"),
                                                       col("未运行/已退出", "r")))
        else:
            pc = s.get("proc_cpu", 0.0)
            extra = ""
            gmem = s.get("proc_gpu_mem") or 0
            if gmem:
                extra = "  GPU %.1f MiB" % (gmem / 1024.0 ** 2)
            lines.append(pad("进程", LW) + "%s(%s)  CPU %s  RSS %.2f GiB%s" % (
                s.get("proc_name", "?"), s.get("proc_pid"),
                col("%5.1f%%" % pc, level(pc)), gib(s.get("proc_rss", 0)), extra))

    peak_bits = []
    for key, label in (("cpu_sum", "CPU"), ("gpu", "GPU")):
        stt = st.get(key)
        if stt:
            peak_bits.append("%s 峰 %.1f%% 均 %.1f%%" % (label, stt["max"], stt["avg"]))
    stt = st.get("mem_used")
    if stt:
        peak_bits.append("内存 峰 %.1f GiB" % gib(stt["max"]))
    if peak_bits:
        lines.append(pad("区间", LW) + col("   ".join(peak_bits), "y"))

    lines.append(pad("", LW) + col("Ctrl+C 结束并输出区间汇总", "y"))
    return lines


def flat_line(s):
    """非 TTY（重定向到文件）时的单行输出。"""
    bits = ["cpu=%.1f%%(整机%.1f%%)" % (s.get("cpu_sum", 0.0), s.get("cpu", 0.0))]
    if s.get("gpu") is not None:
        bits.append("gpu=%.1f" % s["gpu"])
    if s.get("mem_total"):
        bits.append("mem=%.1f/%.1fGiB" % (gib(s["mem_used"]), gib(s["mem_total"])))
    if s.get("vram_used") is not None:
        if s.get("vram_kind") == "nvmap":
            bits.append("gpu_mem=%.1fMiB" % (s["vram_used"] / 1024.0 ** 2))
        else:
            bits.append("vram=%.1fGiB" % gib(s["vram_used"]))
    if s.get("gpu_temp") is not None:
        bits.append("gputemp=%.0fC" % s["gpu_temp"])
    if s.get("power") is not None:
        bits.append("power=%.1fW" % s["power"])
    if "proc_cpu" in s:
        if s.get("proc_dead"):
            bits.append("proc=%s:未运行" % s.get("proc_name", "?"))
        else:
            bits.append("proc=%s:%.1f%%/%.2fGiB" % (
                s.get("proc_name", "?"), s.get("proc_cpu", 0.0), gib(s.get("proc_rss", 0))))
            if s.get("proc_gpu_mem"):
                bits.append("procgpu=%.1fMiB" % (s["proc_gpu_mem"] / 1024.0 ** 2))
    return time.strftime("%H:%M:%S ") + " ".join(bits)


def print_summary(st, t0, t1, backend, host, proc):
    line = "═" * 66
    print()
    print(col(line, "c"))
    print(col("  区间汇总", "c"))
    print(col(line, "c"))
    print("主机    %s   后端 %s" % (host, backend))
    print("时间    %s → %s   时长 %s   采样 %d" % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        time.strftime("%H:%M:%S", time.localtime(t1)),
        hms(t1 - t0), st.n))
    print("─" * 66)
    print(pad("指标", 20) + "%11s%12s%12s" % ("平均", "峰值", "最低"))
    rows = (
        ("cpu_sum", "CPU 总占用（核累加）"),
        ("cpu", "CPU 整机占用"),
        ("cpu_hot", "CPU 最高单核"),
        ("gpu", "GPU 计算占用"),
    )
    cores = st.last.get("cores") or 1
    for key, label in rows:
        d = st.get(key)
        if d:
            note = "   （1 核满载=100%%，满核 %d%%）" % (cores * 100) if key == "cpu_sum" else ""
            print(pad(label, 20) + "%10.1f%%%11.1f%%%11.1f%%" % (
                d["avg"], d["max"], d["min"]) + note)

    total_mem = st.last.get("mem_total", 0)
    d = st.get("mem_used")
    if d:
        print(pad("内存占用", 20) + "%8.1f GiB%8.1f GiB%8.1f GiB   （总 %.1f GiB，均占 %.1f%%）" % (
            gib(d["avg"]), gib(d["max"]), gib(d["min"]),
            gib(total_mem), 100.0 * d["avg"] / total_mem if total_mem else 0.0))

    d = st.get("vram_used")
    if d and st.last.get("vram_kind") == "nvmap":
        mib = 1024.0 ** 2
        print(pad("显存占用", 20) + "%8.1f MiB%8.1f MiB%8.1f MiB   "
                                   "（nvmap 的 GPU 侧分配，同 jtop [GPU MEM]）" % (
            d["avg"] / mib, d["max"] / mib, d["min"] / mib))
    elif d:
        vtotal = st.last.get("vram_total") or 0
        print(pad("显存占用", 20) + "%8.1f GiB%8.1f GiB%8.1f GiB   （总 %.1f GiB）" % (
            gib(d["avg"]), gib(d["max"]), gib(d["min"]), gib(vtotal)))
    elif backend == "tegrastats":
        print(pad("显存占用", 20) + "未统计（Jetson 的 GPU 显存需 root 读 nvmap，用 sudo 运行）")

    for key, label, unit in (("gpu_temp", "GPU 温度", "°C"),
                             ("cpu_temp", "CPU 温度", "°C"),
                             ("soc_temp", "SoC 温度", "°C"),
                             ("power", "功耗", " W")):
        d = st.get(key)
        if d:
            multi_rail = len(st.last.get("power_rails") or {}) > 1
            note = "   （各轨合计，同 jtop ALL）" if key == "power" and multi_rail else ""
            print(pad(label, 20) + ("%10.1f%s%11.1f%s%11.1f%s" % (
                d["avg"], unit, d["max"], unit, d["min"], unit)) + note)

    if proc is not None:
        d = st.get("proc_cpu")
        r = st.get("proc_rss")
        name = st.last.get("proc_name", proc.spec)
        pid = st.last.get("proc_pid")
        if st.last.get("proc_dead"):
            print("进程    %s    未在运行" % name)
        elif d and r:
            g = st.get("proc_gpu_mem")
            gtxt = ""
            if g and g["max"] > 0:
                gtxt = "，GPU 显存 均 %.1f MiB 峰 %.1f MiB" % (
                    g["avg"] / 1024.0 ** 2, g["max"] / 1024.0 ** 2)
            print("进程    %s(%s)  CPU 均 %.1f%% 峰 %.1f%%，RSS 均 %.2f GiB 峰 %.2f GiB%s" % (
                name, pid, d["avg"], d["max"], gib(r["avg"]), gib(r["max"]), gtxt))
    print(col(line, "c"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="终端实时性能监测（CPU / GPU / 内存 / 显存），退出时打印区间汇总",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--interval", type=float, default=1.0,
                    help="采样间隔秒，默认 1.0（最小 0.2）")
    ap.add_argument("-d", "--duration", type=float, default=0.0,
                    help="运行时长秒，0 表示一直跑直到 Ctrl+C（默认 0）")
    ap.add_argument("-p", "--pid", default=None,
                    help="顺带统计某进程的 CPU%%/RSS，可传 PID 或进程名（如 camcap_capture）")
    ap.add_argument("--csv", default=None, help="把每次采样追加写入 CSV")
    ap.add_argument("--plain", action="store_true", help="关闭颜色")
    args = ap.parse_args()

    global USE_COLOR
    live = sys.stdout.isatty()
    USE_COLOR = live and not args.plain
    interval = max(0.2, args.interval)

    host = socket.gethostname()
    mon = Monitor(interval, args.pid)
    backend = mon.gpu.name if mon.gpu is not None else "无"

    screen = Screen()
    st = Stats()
    running = {"go": True}
    state = {"warm": True, "n": 0, "width": 0}

    def on_signal(signum, frame):
        running["go"] = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    writer = reader = None
    if args.csv:
        try:
            new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
            reader = open(args.csv, "a", newline="")
            writer = csv.writer(reader)
            if new_file:
                writer.writerow(["time", "cpu_sum_pct", "cpu_machine_pct", "cores",
                                 "cpu_hot_pct", "gpu_pct",
                                 "mem_used_gib", "mem_total_gib",
                                 "vram_used_gib", "vram_total_gib",
                                 "gpu_temp_c", "cpu_temp_c", "soc_temp_c", "power_w",
                                 "gpu_mem_mib",
                                 "proc_name", "proc_pid", "proc_cpu_pct", "proc_rss_gib"])
        except OSError as exc:
            print("CSV 打开失败：%s" % exc, file=sys.stderr)
            writer = None

    # 预热一个间隔，避免第一帧 CPU% 落在除零区间
    mon.sample()
    time.sleep(interval)

    t0 = time.time()
    deadline = t0 + args.duration if args.duration > 0 else None
    next_tick = t0

    try:
        while running["go"]:
            s = mon.sample()
            st.add(s)
            if writer is not None:
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "%.2f" % s.get("cpu_sum", 0.0), "%.2f" % s.get("cpu", 0.0),
                    "%d" % (s.get("cores") or 0),
                    "%.2f" % s.get("cpu_hot", 0.0),
                    "" if s.get("gpu") is None else "%.2f" % s["gpu"],
                    "%.3f" % gib(s.get("mem_used")), "%.3f" % gib(s.get("mem_total")),
                    "" if s.get("vram_used") is None else "%.3f" % gib(s["vram_used"]),
                    "" if s.get("vram_total") is None else "%.3f" % gib(s["vram_total"]),
                    "" if s.get("gpu_temp") is None else "%.1f" % s["gpu_temp"],
                    "" if s.get("cpu_temp") is None else "%.1f" % s["cpu_temp"],
                    "" if s.get("soc_temp") is None else "%.1f" % s["soc_temp"],
                    "" if s.get("power") is None else "%.2f" % s["power"],
                    "" if s.get("vram_kind") != "nvmap" else "%.1f" % (s["vram_used"] / 1024.0 ** 2),
                    s.get("proc_name", ""), s.get("proc_pid", ""),
                    "" if s.get("proc_cpu") is None else "%.2f" % s["proc_cpu"],
                    "" if s.get("proc_rss") is None else "%.3f" % gib(s["proc_rss"]),
                ])
                reader.flush()
            if live:
                screen.draw(render(s, st, time.time() - t0, interval, backend, host))
            else:
                print(flat_line(s))
            if deadline is not None and time.time() >= deadline:
                break
            next_tick += interval
            wait = next_tick - time.time()
            if wait > 0:
                time.sleep(wait)
            else:
                next_tick = time.time()
    finally:
        mon.close()
        if reader is not None:
            reader.close()

    if not live:
        print_summary(st, t0, time.time(), backend, host, mon.proc)
    else:
        # 原地刷新区先收个尾，再往下打印汇总
        sys.stdout.write("\n")
        print_summary(st, t0, time.time(), backend, host, mon.proc)


if __name__ == "__main__":
    main()
