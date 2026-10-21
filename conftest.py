# pytest 从项目根运行时，把根目录与 src/ 插入 sys.path，
# 使 tests/ 按与 main.py 一致的约定导入：
#   - src/ 下的 messaging / transport（src 布局，以顶层包名导入）；
#   - 根目录下的 modules（modules.ai / modules.script）。
# （pytest 默认只插入各测试文件所在目录，不会插入项目根与 src/。）

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))