# 任务：用"黑帽 + 最小路径"替换画线检测核心

> 执行环境：Ubuntu。先读 `CLAUDE.md` 了解项目结构，再读 `docs/detection_research.md` 了解方案选型依据。
> 本任务实现调研文档第 4 节的推荐方案。完成后由人工查看效果验收。

## 背景一句话

现在 `processor.py` `_step4` 用全局灰度阈值（`gray < 115`）找记号笔画线，皮肤阴影误检、浅笔迹漏检。调研已证明：**黑帽变换响应图 + 鬓角两锚点间 Dijkstra 最小路径**在测试集上平均误差 ≤ 0.5px（实验脚本 `test_image/analyze.py` / `analyze2.py` / `analyze3.py` 可直接运行复现，可作为实现参考）。

## 要做什么

### 1. 重写 `processor.py` 的检测核心（第 4 步）

保留第 1~3 步不动（landmark 上部区域 ∩ 头部分割 = `self._intersection`）。新的第 4 步：

1. **黑帽响应图**：ROI 内对灰度图做 `cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)`，核为椭圆、尺寸 `max(15, int(w * 0.025) | 1)`（w 为图宽）。ROI 外响应置 0。
2. **锚点**：第 1 步已取得 landmark `21`（左鬓角）、`251`（右鬓角）的像素坐标（`_step1` 中 `points` 的首尾元素，需要存为实例属性）。在每个锚点周围的小窗口（建议边长 ≈ 图宽 6%）内取黑帽响应最大的位置作为吸附后的锚点。
3. **最小路径**：代价图 `cost = bh.max() - bh + 1.0`，ROI 外代价设为很大的值（防止路径出界）；用 `skimage.graph.route_through_array(cost, 左锚点, 右锚点, fully_connected=True, geometric=True)` 得到路径（注意它的坐标是 `(row, col)`）。
4. **拒识**：计算路径上的平均黑帽响应；低于阈值（初始值建议 10，做成可调参数）→ 抛 `ValueError("未检测到画线，请确认照片中额头有清晰的手绘标记线")`（中文，与现有错误风格一致）。
5. **生成线 mask**：把路径画成折线并按笔迹宽度膨胀（默认半径 = 现有 `dilation_pct` 逻辑，保留该参数），得到 `self._black_mask`（保持属性名，第 5 步无需大改）。

### 2. 第 5 步与接口适配

- `_step5`（contour / hull mask 生成）逻辑沿用；确认在新 `_black_mask`（一条膨胀后的曲线）上仍能产出合理的闭合 mask。如果"最大轮廓填充"对一条线退化（线本身就是闭合区域边界的一部分），可改为：路径曲线与其上方的 ROI 边界围成闭合多边形后填充——以可视化效果合理为准。
- `/adjust` 接口：`threshold` 参数语义改变。保持 API 字段不变以免改前端结构，但把它重新映射为**拒识阈值**或直接忽略；`dilation_pct` 继续控制线宽膨胀。`templates/index.html` 中滑块的中文说明文字相应更新。
- 第 4 步可视化：在原图上叠加黑帽响应热力图（弱）+ 红色路径（强），便于人工检查路径走位。

### 3. 依赖与环境

- `requirements.txt` 增加 `scikit-image`。
- 注意 `CLAUDE.md` 中的坑：**不要在主进程 import torch**；scikit-image 没有这个问题，可直接在主进程用。
- 服务启动：`./run.sh`，端口 8888。

### 4. 验收脚本（必须写）

新建 `test_image/evaluate.py`：

1. 对 `test_image/` 下每对 `inputN` / `outputN`，跑完整 pipeline（需要 mediapipe 环境；Ubuntu 上 `run.sh` 会装好）；
2. 从 GT 图提取绿线带（`g>150 & r<120 & b<120`，GT 与原图尺寸可能不一致需 resize）；
3. 用 GT 的距离变换计算检测路径每个点的误差，输出 平均 / 中位 / P95 / 最大（px）；
4. 把"原图 + GT 绿线描边 + 检测路径红线"的叠加图存为 `test_image/eval_inputN.jpg`。

### 5. 验收标准

- [ ] `evaluate.py` 在 input1 / input3 上：平均误差 ≤ 2px，最大误差 ≤ 15px；
- [ ] 拒识生效：随便给一张无画线的人脸照片（可临时找/生成），返回"未检测到画线"错误而不是乱画一条线；
- [ ] `./run.sh` 启动后，浏览器上传 input1.png 和 input3.JPG，5 个步骤可视化正常显示，contour / hull mask 可下载且形状合理；
- [ ] `/adjust` 调节 `dilation_pct` 时线宽实时变化，不报错；
- [ ] 不破坏现有约定：单实例状态缓存、`/adjust` 只重跑 4~5 步、主进程不 import torch。

### 6. 明确不做

- 不动第 1~3 步的 ROI 逻辑；
- 不上深度学习模型（那是后备方案，见调研文档 3.4）；
- 不做多用户会话改造。

### 7. 交付物

- 修改后的 `processor.py`、`app.py`（如需）、`templates/index.html`（滑块文案）、`requirements.txt`；
- 新增 `test_image/evaluate.py` 及评估输出图；
- 在 PR/提交说明里贴出 evaluate.py 的误差数字和叠加图，供人工验收。
