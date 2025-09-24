# 如何让渲染画面停留久一点

## 问题描述
执行 `bash collect_data.sh turn_switch demo_randomized_rm65b 0` 时，渲染画面显示时间太短，希望让画面停留更久。

## 解决方案

### 方法1：修改渲染频率（推荐）

在任务配置文件中增加 `render_freq` 的值，让渲染的间隔更大，从而让每次渲染的画面显示更久。

**修改步骤：**

1. 打开 `task_config/demo_randomized_rm65b.yml`
2. 将 `render_freq: 5` 改为更大的值，例如 `render_freq: 20` 或 `render_freq: 50`

```yaml
# 修改前
render_freq: 5

# 修改后（推荐）
render_freq: 20
```

**说明：**
- `render_freq = 5` 表示每5帧渲染一次
- `render_freq = 20` 表示每20帧渲染一次
- 值越大，渲染间隔越大，画面显示时间越长
- 但如果设置太大，可能影响任务执行的流畅性

### 方法2：在代码中添加延时

如果需要更精确的控制，可以修改 `_base_task.py` 中的渲染代码，添加延时。

**在第896行 `self.viewer.render()` 后添加延时：**

```python
self.viewer.render()
import time
time.sleep(0.1)  # 增加0.1秒延时
```

**在第1469行 `self.viewer.render()` 后添加延时：**

```python
self.viewer.render()
import time
time.sleep(0.1)  # 增加0.1秒延时
```

**在第1493行 `self.viewer.render()` 后添加延时：**

```python
self.viewer.render()
import time
time.sleep(0.1)  # 增加0.1秒延时
```

## 推荐设置

根据你的需求，建议首先尝试修改 `render_freq`：

```yaml
render_freq: 15  # 或者 20-30，根据需要调整
```

这个设置可以在不影响代码的情况下，让画面显示时间显著增加。

## 注意事项

1. 渲染频率设置太高可能导致任务执行变慢
2. 如果需要更精细的控制，可以考虑添加延时代码
3. 延时时间可以根据需要调整，0.05-0.2秒通常比较合适