# perf_mon.py — 终端实时性能监测 使用说明

一个单文件、纯标准库的终端性能监视器：原地刷新显示 **CPU / GPU 计算 / 内存 / 显存 / 温度 / 功耗**，
Ctrl+C 结束后打印这段时间的 **平均 / 峰值 / 最低** 汇总。

| 项 | 说明 |
|---|---|
| 依赖 | 无。Python 3.7+ 标准库即可（板上 3.10、本机 3.10 都验证过） |
| 本机路径 | `/home/sky/sunrise/scripts/perf_mon.py` |
| 板端路径 | `/home/zndp/tools/perf_mon.py`（已上传，可执行） |
| 适用平台 | x86 + NVIDIA 独显、Jetson（Orin / Xavier 等 L4T 平台） |

---

## 1. 快速开始

```bash
# 本机（RTX 4090）
cd /home/sky/sunrise
./scripts/perf_mon.py

# 板上（Jetson，看 GPU 显存需要 root）
sudo ~/tools/perf_mon.py -p camcap_capture
```

跑起来后在当前窗口原地刷新，**Ctrl+C 退出并打印区间汇总**。

---

## 2. 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i, --interval` | `1.0` | 采样间隔（秒），最小 `0.2` |
| `-d, --duration` | `0` | 运行时长（秒），`0` = 一直跑直到 Ctrl+C |
| `-p, --pid` | 无 | 额外追踪一个进程，可传 **PID** 或 **进程名**（如 `camcap_capture`） |
| `--csv <path>` | 无 | 每次采样追加写一行 CSV，文件不存在/为空时自动写表头 |
| `--plain` | 关 | 关闭 ANSI 颜色（日志重定向时也会自动关） |
| `-h, --help` | — | 打印帮助 |

### 常用组合

```bash
# 盯采集节点跑 60 秒，同时落表
sudo ~/tools/perf_mon.py -p camcap_capture -d 60 --csv ~/camcap_run.csv

# 0.5 秒采样，看瞬时毛刺
./scripts/perf_mon.py -i 0.5

# 包住一次完整评测：跑完自动出这段的均值/峰值
sudo ~/tools/perf_mon.py -p camcap_capture -d 180

# 只按 PID 盯（进程名会变的重命名场景）
./scripts/perf_mon.py -p 11341

# 无颜色，方便贴进文档
./scripts/perf_mon.py --plain
```

> `-p` 传进程名时：先用 `pgrep -x` 精确匹配进程名，再用 `pgrep -f` 匹配完整命令行；
> 两种方式都会排除脚本自身和父进程，不会把自己匹配进去。
> 进程没起 / 中途重启 / 退出后再启动，都会**自动重新捕获**，不需要重启脚本。

---

## 3. 画面字段含义

```
perf_mon · ubuntu · tegrastats · 间隔 1.0s · 已运行 00:12 · 采样 12
CPU 总占用                   171.6%  ██████░░░░░░░░░░░░░░  满核 1200%（12 核累加）
CPU 整机占用                  14.3%  ██░░░░░░░░░░░░░░░░░░  12 核 · 最忙核 100.0% · load 1.58
GPU 计算                      82.0%  ████████████████░░░░  EMC 12%
内存            7.1/61.4 GiB  11.5%  ██░░░░░░░░░░░░░░░░░░  swap 0.0/30.7 GiB
显存               31.4 MiB  0.05%  nvmap 的 GPU 侧分配（同 jtop [GPU MEM] 合计）
温度         CPU 67.1°C  GPU 66.5°C  SoC 67.1°C    功耗 37.6 W（GPU_SOC 33.6 + CPU_CV 4.0）
进程         camcap_capture(11341)  CPU 134.8%  RSS 1.21 GiB  GPU 12.4 MiB
区间         CPU 峰 180.0% 均 171.6%   GPU 峰 98.0% 均 56.2%   内存 峰 7.1 GiB
             Ctrl+C 结束并输出区间汇总
```

| 行 | 口径 |
|---|---|
| **CPU 总占用** | **各核累加**：1 核满载 = 100%，12 核满载 = 1200%（`top` 的 %CPU 口径）。进度条按满核归一 |
| **CPU 整机占用** | 整机占比 0–100%（`总占用 ÷ 核数`），看"机器用了多少"用这行 |
| **最忙核** | 单核占用最大值，0–100%，用来判断是否有线程把一颗核跑满 |
| **load** | `/proc/loadavg` 的 1 分钟负载 |
| **GPU 计算** | GPU 计算单元占用率。x86 取 `nvidia-smi utilization.gpu`；Jetson 取 `tegrastats GR3D_FREQ` |
| **内存** | `MemTotal − MemAvailable`（比 `free` 的 used 更贴近实际可用），附 swap |
| **显存** | 独显：`nvidia-smi memory.used / total`（GiB + 进度条）；Jetson：**nvmap 的 GPU 侧真实分配**（MiB，同 jtop 进程表 `[GPU MEM]` 合计）。注意 Jetson 是统一内存，不存在"独立显存" |
| **温度** | CPU / GPU / SoC（`tj` 结温）。平台缺哪项就不显示哪项 |
| **功耗** | 独显：`nvidia-smi power.draw` 整卡功耗；Jetson：**所有电源轨累加**，与 jtop 的 `ALL` 同口径，括号里给各轨明细 |
| **进程** | `-p` 追踪的进程：CPU%（同样按单核累加，可 >100%）、RSS、GPU 显存 |
| **区间** | 本次运行到当前为止的峰值/均值（CPU 用核累加口径） |

颜色：<50% 绿、<80% 黄、≥80% 红。

---

## 4. 退出汇总

Ctrl+C（或 SIGTERM）后打印：

```
══════════════════════════════════════════════════════════════════
  区间汇总
══════════════════════════════════════════════════════════════════
主机    ubuntu   后端 tegrastats
时间    2026-11-05 14:08:12 → 14:08:16   时长 00:04   采样 5
──────────────────────────────────────────────────────────────────
指标                         平均          峰值          最低
CPU 总占用（核累加）       14.1%       14.9%       13.0%   （1 核满载=100%，满核 1200%）
CPU 整机占用               1.2%        1.2%        1.1%
CPU 最高单核             100.0%      100.0%      100.0%
GPU 计算占用              97.2%       99.0%       82.0%
内存占用                 7.1 GiB     7.1 GiB     7.1 GiB   （总 61.4 GiB，均占 11.5%）
显存占用                31.4 MiB    31.4 MiB    31.4 MiB   （nvmap 的 GPU 侧分配，同 jtop [GPU MEM]）
GPU 温度                  66.8°C       67.0°C       66.5°C
CPU 温度                  67.0°C       67.1°C       66.9°C
SoC 温度                  67.1°C       67.1°C       67.0°C
功耗                      37.6 W       37.6 W       37.6 W   （各轨合计，同 jtop ALL）
进程    camcap_capture(11341)  CPU 均 133.3% 峰 135.9%，RSS 均 3.62 GiB 峰 3.62 GiB，GPU 显存 均 12.4 MiB 峰 12.4 MiB
══════════════════════════════════════════════════════════════════
```

同一份数据也可以直接 `--csv` 落盘，事后用 pandas / Excel 画曲线。

---

## 5. CSV 列

```
time                采样时间
cpu_sum_pct         CPU 核累加%
cpu_machine_pct     CPU 整机%
cores               核数
cpu_hot_pct         最忙单核%
gpu_pct             GPU 计算%
mem_used_gib        已用内存 GiB
mem_total_gib       总内存 GiB
vram_used_gib       独显显存 GiB（Jetson 上为空）
vram_total_gib      显存总量 GiB（Jetson 上为空）
gpu_temp_c          GPU 温度
cpu_temp_c          CPU 温度
soc_temp_c          SoC 温度
power_w             功耗 W
gpu_mem_mib         Jetson nvmap GPU 显存 MiB（独显上为空）
proc_name/proc_pid  被追踪进程
proc_cpu_pct        该进程 CPU%（可 >100%）
proc_rss_gib        该进程 RSS GiB
```

表头只在文件不存在或为空时写入，重复运行会**追加**。

---

## 6. 平台差异 / 后端选择

程序启动时自动挑后端，规则按优先级：

| 平台 | 判定 | GPU 占用 | 显存 | 功耗 |
|---|---|---|---|---|
| Jetson | 存在 `/etc/nv_tegra_release` | `tegrastats GR3D_FREQ` | `nvmap`（需 root） | 各电源轨累加 |
| x86 + 独显 | `nvidia-smi` 实测能返回数字 | `utilization.gpu` | `memory.used` | `power.draw` |
| 其它 | 兜底 | `/sys/class/devfreq/gpu/load` | — | — |

> 注意：Jetson 上**也有** `nvidia-smi`，但 `--query-gpu=utilization.gpu` 返回 `[N/A]`，
> 所以程序会实际试一次，拿不到数字就改用 `tegrastats`。

---

## 7. 常见问题

**Q: 显存那行显示 `-- 需 root 读 nvmap（用 sudo 运行）`？**
Jetson 的 GPU 显存统计在 debugfs（`/sys/kernel/debug/nvmap/iovmm/maps`），只有 root 能读。
用 `sudo` 运行即可；不加也只是这一行没有，其它指标照常。jtop 能显示是因为它背后有 root 服务在采。

**Q: `GPU 计算` 显示 `--`？**
说明后端启动失败或该平台没有可用的 GPU 占用率接口。Jetson 上首次启动会等 `tegrastats`
吐出第一行（最多约 2 秒），之后才会有数据。

**Q: `最忙核` 一直是 100%，是不是算错了？**
不是。Jetson 上常见于 USB 取流 / 编解码线程：`/proc/stat` 的 idle 计数一秒内不增长，
说明这颗核真的跑满了；但整机只有 14% 是因为 12 核在分摊。

**Q: 功耗和 jtop 差 0.x W？**
两边都是"所有电源轨之和"，差异来自采样时刻不同（各轨是 mW 级实时值）。

**Q: 进程显示"未运行/已退出"？**
进程确实不在。进程稍后起来会自动接上；已经结束就会一直显示未运行。

**Q: 重定向到文件后画面没了？**
非 TTY 时自动退化成"每秒一行文本 + 汇总"，避免满屏 ANSI 控制符：

```
14:19:55 cpu=10.9%(整机0.9%) gpu=0.0 mem=6.0/61.4GiB gpu_mem=31.4MiB power=9.5W proc=...:2.0%/0.21GiB
```

**Q: 板上每次都要输密码？**
可以只给这一条命令配免密：

```bash
echo 'zndp ALL=(root) NOPASSWD: /home/zndp/tools/perf_mon.py' | sudo tee /etc/sudoers.d/perf_mon
sudo chmod 440 /etc/sudoers.d/perf_mon
```

---

## 8. 典型用途

```bash
# 1) 相机采集节点的负载画像（含 GPU 编码、USB 取流线程）
sudo ~/tools/perf_mon.py -p camcap_capture --csv ~/camcap_load.csv

# 2) 评测/压测期间全程记录，结束后看峰值是否触到温度/功耗墙
sudo ~/tools/perf_mon.py -d 600 --csv ~/soak.csv

# 3) 对比两种配置的性能开销
sudo ~/tools/perf_mon.py -p camcap_capture -d 60 --csv ~/cfgA.csv
```
