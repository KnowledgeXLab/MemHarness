python -c "import torch; print(torch.__version__); print('cuda', torch.version.cuda); print('avail', torch.cuda.is_available())"
pip show torch

# 注意：不要先 `from torch._inductor.kernel import ...`。
# `torch._inductor.kernel` 的 __init__ 会先加载 mm.py → cpp_gemm_template，
# 在未完成 lowering 初始化时与 quantized_lowerings 形成循环导入（CppGemmTemplate）。
# 应先让 lowering 完整加载，再访问 kernel 子模块。
python -c "
import torch
import torch._inductor.lowering  # noqa: F401 — 必须先于 kernel
from torch._inductor.kernel import mm_common
print('mm_common ok, has persistent_mm_grid:', hasattr(mm_common, 'persistent_mm_grid'))
"

python -c "
import torch
fn = torch.compile(lambda x: x + 1, backend='inductor')
print(fn(torch.tensor(1.)))
"
