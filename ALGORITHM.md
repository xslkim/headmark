# HeadMark 算法文档

本文档详细说明 HeadMark 当前使用的发际线 / 马克笔标记线检测与蒙板生成算法。

> 适用代码版本：`processor.py` + `doubao_vision.py`（VLM 引导版）。
> 与早期 CLAUDE.md 中"近黑像素阈值 + 膨胀"的描述相比，核心检测逻辑已升级为
> **VLM 语义定位 + 局部 ridge 跟随**，旧的纯阈值算法现已降级为 fallback 分支。

---

## 1. 总览

HeadMark 接收一张植发场景的人脸照片（额头/头皮上用黑色马克笔画了一条预期发际线），
输出两种蒙板（mask）供扩散模型使用：

- **contour mask**：沿马克笔线本身的细蒙板（描边）。
- **hull mask**：覆盖标记区域的凸包蒙板（填充）。

整个流程分为 5 个步骤。其中 **步骤 1–3 在一次上传中只运行一次并被缓存**，
调节 `threshold` / `dilation` 时只重跑 **步骤 4–5**（`/adjust` 接口）。

```
上传图片
  │
  ├─ Step1  人脸关键点 → 额头上半区多边形 mask        ┐
  ├─ Step2  头部分割 mask                            │ 缓存，
  ├─ Step3  上半区 ∩ 头部 = ROI 交集 mask            ┘ /adjust 不重跑
  │
  ├─ VLM    Doubao Vision 沿马克笔线采样折线（语义先验）
  │
  ├─ Step4  在折线附近的窄带内用 DP 跟随真实墨迹脊线   ┐ /adjust
  └─ Step5  生成 contour / hull 蒙板                  ┘ 重跑
```

两条检测路径：

| 路径 | 触发条件 | 核心思想 |
|------|----------|----------|
| **VLM 引导**（首选） | 提供了 Doubao API Key 且模型返回 ≥2 个点 | 用 VLM 折线做语义先验，黑帽变换 + 动态规划锁定真实墨迹 |
| **本地 fallback** | 无 API Key / VLM 失败 / 折线退化 | 在 ROI 内做自适应阈值 + 局部暗度检测 |

---

## 2. 模型与依赖

| 用途 | 首选 | 回退 |
|------|------|------|
| 人脸关键点 | MediaPipe `face_landmarker.task` | —（无脸则报错） |
| 头部分割 | `head-segmentation`（PyTorch，子进程隔离运行） | MediaPipe `selfie_segmenter.tflite` |
| 标记线语义定位 | Doubao Vision（火山引擎方舟，`doubao-seed-1-6-vision-250815`） | 本地阈值算法 |

**GPU 冲突隔离**：`head-segmentation`（PyTorch）和 MediaPipe 会争抢 GPU 导致崩溃，
因此头部分割在**独立子进程**中跑（`_run_head_seg_subprocess`），通过临时 `.npy`
文件做进程间 I/O，结束后自动清理。

---

## 3. 逐步算法

### Step 1 — 人脸关键点 → 额头上半区多边形

`_step1()`

1. MediaPipe FaceLandmarker 检测单张人脸（`num_faces=1`，置信度 0.5）。
2. 取一组沿额头上沿的关键点索引：
   `LANDMARK_INDICES = [21, 68, 104, 69, 108, 151, 337, 299, 333, 298, 251]`
   （从左额角经眉上、额头中线到右额角）。
3. 把这些点向左右两侧延伸到图像边缘，再连上图像顶部两角，
   闭合成一个覆盖**额头以上整个上半区**的多边形。
4. `fillPoly` 得到 `_upper_mask`（白=上半区）。

可视化：上半区高亮、其余压暗 0.3，画出关键点编号和连线。

> 作用：把搜索范围从整图缩到"额头及以上"，排除五官、下巴等区域。

### Step 2 — 头部/人像分割

`_step2()`

- 若 `head-segmentation` 可用：子进程跑 `HumanHeadSegmentationPipeline.predict`，
  `seg_map > 0` 二值化为 `_head_mask`。
- 否则：MediaPipe Selfie Segmentation，`confidence > 0.5` 二值化。

可视化：头部区域保留、其余压暗，红色描出头部轮廓。

> 作用：得到"属于人头/人像"的像素，下一步用来裁掉背景里误入上半区的东西。

### Step 3 — ROI 交集

`_step3()`

```
_intersection = _upper_mask AND _head_mask
```

即"额头上半区"与"头部"的交集，这才是马克笔线**可能出现的真实区域**。
结果缓存为 `_intersection`，作为后续检测的 ROI。

### VLM — Doubao Vision 语义定位（`_run_vlm` + `doubao_vision.detect_marker`）

每次上传调用一次。流程：

1. 把 BGR 图编码为 JPEG → base64，连同中文提示词发给方舟 Vision 模型。
2. 提示词要求模型**沿马克笔线从最左到最右均匀采样约 15 个点**，
   严格返回 JSON：`{"has_marker": true, "points": [[x,y], ...]}`，
   坐标为 0–1 的归一化比例；无标记线则 `{"has_marker": false}`。
3. 解析（容错 markdown 代码块包裹），坐标裁剪到 [0,1]，少于 2 点视为无标记。
4. 结果缓存到 `self._vlm_points`，并返回状态字符串供前端显示：
   `ok` / `disabled` / `module_missing` / `api_error` / `no_marker_detected` / `no_points`。

认证：API Key（`ark-` 前缀）走 `env DOUBAO_API_KEY` 或 `apikey.txt`，作为 Bearer Token。
模型 ID **必须带版本号**（不带版本会 404）。

> 关键点：VLM 只提供**语义先验**——它知道"线大概在哪、怎么走"，但其 y 坐标在
> 多次调用间会上下抖动 ±15–20 px，**不能直接信它的坐标**。真正的定位交给 Step 4。

### Step 4 — 标记线检测

`_step4()` 按是否有 VLM 折线分两条路径。

#### 4A. VLM 引导路径（首选，`_step4_vlm_guided`）

核心：**黑帽变换提取墨迹 + 动态规划在窄带内跟随脊线**。

1. **确定 x 范围**：把归一化折线点换算成像素 `px`，取 `x_lo..x_hi`
   （跨度 < 10 px 则退化，返回 `None` 走 fallback）。

2. **黑帽变换（Black-hat）**：
   ```
   blackhat = morphologyEx(gray, MORPH_BLACKHAT, ellipse(15×15))
   ```
   黑帽 = 闭运算结果 − 原图，能**高亮比结构元更细的暗线**，
   无论背景是浅色皮肤还是深色头发都有响应。
   这正是解决"马克笔和背景灰度相近"难题的关键。

3. **构造先验曲线**：对 `x_lo..x_hi` 每一列，用 `np.interp` 在 VLM 折线上插值出
   先验 y（`prior_y`）。搜索带半高 `band_half = max(10, h·0.030)`。

4. **动态规划跟随脊线**（`_dp_follow_ridge`，从左到右最小代价路径）：
   - 每列在 `prior_y ± band_half` 的偏移范围内取候选行；
   - **节点代价** `= −墨迹强度 + 0.8·|偏移|`
     （墨迹越强代价越低；离 VLM 先验越远代价略增）；
   - **转移代价** `= 3.0·|相邻列行差|`（惩罚不平滑、抖动）；
   - 前向 DP 填表 + 回溯，得到每列的绝对 y。
   - 权重常量：`_DP_PRIOR_WEIGHT=0.8`、`_DP_SMOOTH_WEIGHT=3.0`。

   > 这样既锁定真实墨迹、又对发缘噪声和 VLM 抖动鲁棒。

5. **平滑**：对 DP 路径做半窗 9 px 的移动平均（`_smooth_path`），裁剪到图像内。
   得到拟合曲线 `_fitted_curve`（Nx2 像素点）。

6. **生成 `_black_mask`**：把曲线用 `polylines` 按厚度 `radius=w·dilation%/100`
   画成带状蒙板（`thickness = radius·2+1`）。

可视化：搜索带压暗 0.5，墨迹（`blackhat > 20`）染琥珀色，红色画出拟合曲线。

#### 4B. 本地 fallback 路径（`_step4_fallback`）

无 VLM 时，在 `_intersection` ROI 内组合两种检测：

1. **自适应绝对阈值**（皮肤上的马克笔）：
   - 取 ROI 内灰度 75 百分位作为肤色基准 `skin_level`；
   - `adaptive_thresh = skin_level · 0.58`；
   - `effective_threshold = clamp(min(用户threshold, adaptive_thresh), 下限40)`；
   - `dark_abs = ROI 内 gray < effective_threshold`。
2. **局部暗度检测**（深色头发上的马克笔）：
   - `gray_blur = GaussianBlur(gray, 25)`；
   - `diff = gray_blur − gray`，`dark_local = ROI 内 diff > 20`。
3. `combined = dark_abs OR dark_local`，按 `dilation%` 椭圆核膨胀，
   再与 ROI 相与，得到 `_black_mask`。

可视化：ROI 外压暗，检测像素染红。

### Step 5 — contour / hull 蒙板生成

`_step5()`

若 `_black_mask` 有效像素 < 50，直接返回空蒙板。

#### 5A. VLM 引导路径（`_fitted_curve` 存在）

马克笔带已经是一条干净的单曲线，无需复杂清理：
- `contour_mask = _black_mask`（曲线带本身）；
- `hull_mask` = 对带状像素求**凸包**并填充。

#### 5B. fallback 路径（基于连通域 + 形状评分）

1. **形态学清理**：开运算（3×3）去噪 → 闭运算（15×15）连接断点。
2. `findContours` 取外轮廓。
3. **形状评分**（`_score_contour`）选主体：
   ```
   score = area · min(elongation, 10) · orientation_bonus
   ```
   - `elongation = 长边/短边`：奖励**水平细长**形状（马克笔线）；
   - `orientation_bonus = 1.5`（宽≥高，即水平）否则 1.0；
   - 面积 < 20 直接判 0（过滤发根噪点、阴影）。
4. **主体 + 伴随碎片**：选最高分轮廓为主体；其余轮廓只要
   分数 ≥ 主体的 25% **且** 垂直中心与主体相差 ≤ 图高 10%，就一并纳入
   （把同一条线的断段拼回来）。填充为 `contour_mask`。
5. `hull_mask` = 对所有清理后像素求凸包并填充。

两条路径都返回 contour / hull 的可视化叠加图。

---

## 4. 关键参数一览（`processor.py` 顶部）

| 常量 | 值 | 含义 |
|------|----|------|
| `LANDMARK_INDICES` | 11 个索引 | 额头上沿关键点 |
| `_VLM_BAND_FRAC` | 0.030 | DP 搜索带半高（占图高比例） |
| `_BLACKHAT_KSIZE` | 15 | 黑帽结构元尺寸（px），决定可检测线宽上限 |
| `_BLACKHAT_THRESH` | 20 | 黑帽响应阈值（可视化用） |
| `_DP_PRIOR_WEIGHT` | 0.8 | DP 偏离 VLM 先验的代价权重 |
| `_DP_SMOOTH_WEIGHT` | 3.0 | DP 相邻列行差（平滑）代价权重 |
| `_DP_SMOOTH_WINDOW` | 9 | DP 路径移动平均半窗（px） |
| `_LOCAL_DARK_DELTA` | 20 | fallback 局部暗度最小差值 |
| `_LOCAL_BLUR_KSIZE` | 25 | fallback 局部暗度高斯模糊核 |

运行时可调参数（`/upload`、`/adjust`）：
- `threshold`（默认 115，范围 0–255）：fallback 绝对阈值。
- `dilation_pct`（默认 1.0，范围 0–5）：线/带厚度，按图宽百分比。

---

## 5. 缓存与接口对应

| 接口 | 重跑步骤 | 说明 |
|------|----------|------|
| `POST /upload` | Step1–3 + VLM + Step4–5 | 完整流程，缓存 1–3 与 VLM 折线 |
| `POST /adjust` | 仅 Step4–5 | 复用缓存，快速调节阈值/厚度 |
| `GET /download_mask?type=contour\|hull` | — | 下载 PNG 蒙板 |

> 因此 VLM **只在上传时调用一次**，调节滑块不会重复消耗 API 配额。

---

## 6. 设计要点小结

1. **VLM 负责"在哪"，CV 负责"精确到哪"**：大模型给语义先验，黑帽 + DP 给亚区域精度，
   规避了"马克笔灰度≈背景"和"VLM 坐标抖动"两个核心难题。
2. **黑帽变换**是对"线比背景暗、但绝对灰度不一定低"场景的关键武器，
   对浅肤色和深发色统一有效。
3. **动态规划**在保证连续平滑的前提下贴合真实墨迹，比逐列取最暗点鲁棒得多。
4. **多重降级**：head-seg→selfie-seg、VLM→本地阈值，任一环节失效都能继续出结果。
