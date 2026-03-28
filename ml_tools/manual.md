# ml_tools 用户手册（nnof）

## 1. 功能概览

`nnof` 用于训练或测试 ML-KEDF / ML-EXX 模型。

## 2. 编译与运行

代码通过 CMake 构建，核心依赖：

- C++ 编译器
- LibTorch
- libnpy

> 若使用LibTorch-2.1.0及以上版本，需要支持C++ 17标准的编译器；否则需要支持C++ 14标准的编译器。

编译时，在ml_tools目录下运行

```bash
cmake -B build \
  -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
  -Dlibnpy_INCLUDE_DIR=/path/to/libnpy
cmake --build build -j
```

即可生成可执行文件：

- `build/nnof`

运行时，需要在工作目录准备输入文件 `nnINPUT`，创建用于存储模型的`model`文件夹，之后在工作目录运行 `/path/to/nnof/nnof`，目前仅支持串行。
训练完成后，所有存储的模型都保存在`model`文件夹中，名称为`net{n}.pt`，其中`n`为对应的训练epoch数，一般选最后输出的模型用于后续推断。

## 3. `nnINPUT` 输入参数说明

### 3.1 重要顺序约束

以下参数存在先后依赖关系：

- 先设置 `ntrain`，再设置 `train_dir`、`train_cell`、`train_a`。
- 先设置 `nvalidation`，再设置 `validation_dir`、`validation_cell`、`validation_a`。
- 先设置 `nkernel`，再设置所有长度为 `nkernel` 的数组参数（如 `kernel_type`、`gammanl`、`chi_xi` 等）。

### 3.2 训练数据与任务类型

#### fftdim

- **Type**: Integer
- **Description**: 三维 FFT 网格边长，训练集中单个构型包含 `fftdim^3` 个网格点。
- **Default**: 0

#### nbatch

- **Type**: Integer
- **Description**: 每个batch包含的格点数，目前程序中默认每个batch包含`fftdim^3`个格点，此参数未实际使用。
- **Default**: 0

#### ntrain

- **Type**: Integer
- **Description**: 训练集构型数。程序据此分配 `train_dir`、`train_cell`、`train_a` 的数组长度。
- **Default**: 1

#### nvalidation

- **Type**: Integer
- **Description**: 验证集构型数。大于 0 时才读取验证集目录及对应标签。
- **Default**: 0

#### train_dir

- **Type**: String 列表
- **Description**: 训练集构型目录列表，长度必须等于 `ntrain`，每个目录下需要存在保存描述子、训练目标的`.npy`文件。
- **Default**: 无

#### train_cell

- **Type**: String 列表
- **Description**: 测试集中每个构型的原胞类型。目前支持 `sc`、`fcc`、`bcc`。
- **Default**: 无

#### train_a

- **Type**: Real 列表
- **Description**: 测试集中每个构型的晶格常数，长度为 `ntrain`，单位为Bohr。
- **Default**: 无

#### validation_dir

- **Type**: String 列表
- **Availability**: `nvalidation > 0`
- **Description**: 验证集构型目录列表，长度为 `nvalidation`，每个目录下需要存在保存描述子、训练目标的`.npy`文件。
- **Default**: 无

#### validation_cell

- **Type**: String 列表
- **Availability**: `nvalidation > 0`
- **Description**: 验证集中每个构型的原胞类型，支持 `sc`、`fcc`、`bcc`。
- **Default**: 无

#### validation_a

- **Type**: Real 列表
- **Availability**: `nvalidation > 0`
- **Description**: 测试集中每个构型的晶格常数，单位为Bohr。
- **Default**: 无

#### energy_type

- **Type**: String
- **Description**: 选择训练任务。
  - `kedf`: 训练动能泛函，读取 `enhancement.npy`，势标签为 `pauli.npy`
  - `exx`: 训练精确交换能，读取 `enhancement_x.npy`，势标签为 `v_pbe_x.npy`
- **Default**: `kedf`

### 3.3 损失函数与优化控制

#### loss

- **Type**: String
- **Description**: 损失函数类型。
  - `energy`: 仅能量项
  - `potential`: 仅势项
  - `both`: 势项 + 能量项
  - `both_new`: 势项 + 加权能量项（仅测试用）

  若打开feg_limit选项，loss会加上自由电子气极限修正的正则项。
- **Default**: `both`

#### nepoch

- **Type**: Integer
- **Description**: 训练总 epoch 数。
- **Default**: 1000

#### lr_start

- **Type**: Real
- **Description**: SGD 初始学习率。
- **Default**: 0.01

#### lr_end

- **Type**: Real
- **Description**: 学习率衰减目标值。
- **Default**: 1e-4

#### lr_fre

- **Type**: Integer
- **Description**: 学习率更新频率（每 `lr_fre` 个 epoch 更新一次）。
- **Default**: 5000

#### exponent

- **Type**: Real
- **Description**: 密度权重指数参数。`both_new` 模式下用于权重 `rho^(exponent/3)`。仅作测试使用。
- **Default**: 5.0

#### coef_e

- **Type**: Real
- **Description**: 损失函数中能量项系数。
- **Default**: 1.0

#### coef_p

- **Type**: Real
- **Description**: 损失函数中势能项系数。
- **Default**: 1.0

#### coef_feg_e

- **Type**: Real
- **Description**: feg_limit为1时，损失函数中正则项系数。
- **Default**: 1.0

#### dump_fre

- **Type**: Integer
- **Description**: 模型保存频率。每隔 `dump_fre` 个 epoch 保存一次模型。
- **Default**: 1

#### print_fre

- **Type**: Integer
- **Description**: 日志打印频率。每隔 `print_fre` 个 epoch 在屏幕输出中打印一次训练信息。
- **Default**: 1

### 3.4 设备、模式与网络结构

#### device_type

- **Type**: String
- **Description**: 计算设备类型。
  - `cpu`: 使用 CPU 训练
  - `gpu`: 使用 GPU 训练，GPU 不可用时会自动回退到 CPU。编译时需要链接支持CUDA的LibTorch。
- **Default**: `gpu`

#### check_pot

- **Type**: Boolean
- **Description**: 用于测试势能正确性。
  - `0` 或 `false`: 正常训练模式
  - `1` 或 `true`: 势测试模式，读取网络 `net.pt` 和训练集的数据后输出增强因子和势能。
- **Default**: `false`

#### nnode

- **Type**: Integer
- **Description**: 隐藏层神经元个数。
- **Default**: 10

#### nlayer

- **Type**: Integer
- **Description**: 隐藏层个数，目前隐藏层固定为3个，此参数不起作用。
- **Default**: 3

### 3.5 自由电子气极限（FEG）约束参数

#### feg_limit

- **Type**: Integer
- **Description**: 引入自由电子气极限（free electron gas, FEG）约束的方式。
  - `0`: 不考虑 FEG 约束
  - `1`: 对网络输出$F^{\mathrm{NN}}$做后处理$F_\theta^{\mathrm{NN}}=F^{\mathrm{NN}} - F^{\mathrm{NN}}|_{\mathrm{FEG}} + 1$，并在损失函数中加入 $(F^{\mathrm{NN}}|_{\mathrm{FEG}} - 1)^2$项
  - `2`: 不做后处理，仅在损失函数中加入 $(F^{\mathrm{NN}}|_{\mathrm{FEG}} - 1)^2$项
  - `3`: 对网络输出$F^{\mathrm{NN}}$做后处理$F_\theta^{\mathrm{NN}}=\mathrm{softplus}(F^{\mathrm{NN}} - F^{\mathrm{NN}}|_{\mathrm{FEG}} + \ln(e-1))$，并在损失函数中加入 $(F^{\mathrm{NN}}|_{\mathrm{FEG}} - \ln(e-1))^2$项。其中$\mathrm{softplus}(x) = \ln(1+e^x)$。
- **Default**: 0

#### change_step

- **Type**: Integer
- **Availability**: `feg_limit = 3`
- **Description**: 在`change_step`步数后再做FEG后处理，以提升训练稳定性。实际训练不需要此参数，仅测试用。
- **Default**: 0

### 3.6 半局域描述子开关

#### gamma

- **Type**: Boolean
- **Description**: 是否启用局域描述子 $\gamma(\mathbf{r})$。
  $\gamma(\mathbf{r})$ 定义为
  $$
  \gamma(\mathbf{r}) = \left(\frac{\rho(\mathbf{r})}{\rho_0}\right)^{1/3}
  $$
  其中 $\rho_0$ 是平均电子密度。
- **Default**: `false`

#### p

- **Type**: Boolean
- **Description**: 是否启用半局域描述子 $p(\mathbf{r})$。
  $$
  p(\mathbf{r})=\frac{|\nabla \rho(\mathbf{r})|^2}{\left[2(3\pi^2)^{1/3}\rho^{4/3}(\mathbf{r})\right]^2}
  $$
- **Default**: `false`

#### q

- **Type**: Boolean
- **Description**: 是否启用半局域描述子 $q(\mathbf{r})$。
  $$
  q(\mathbf{r})=\frac{\nabla^2\rho(\mathbf{r})}{4(3\pi^2)^{2/3}\rho^{5/3}(\mathbf{r})}
  $$
- **Default**: `false`

#### tanhp

- **Type**: Boolean
- **Description**: 是否启用半局域描述子 $\tilde{p}(\mathbf{r})$。
  $$
  \tilde{p}(\mathbf{r})=\tanh\left(\chi_p\,p(\mathbf{r})\right)
  $$
- **Default**: `false`

#### tanhq

- **Type**: Boolean
- **Description**: 是否启用半局域描述子 $\tilde{q}(\mathbf{r})$。
  $$
  \tilde{q}(\mathbf{r})=\tanh\left(\chi_q\,q(\mathbf{r})\right)
  $$
- **Default**: `false`

#### chi_p

- **Type**: Real
- **Description**: 超参数 $\chi_p$，用于控制 $\tilde{p}(\mathbf{r})=\tanh(\chi_p p(\mathbf{r}))$ 的分布。
- **Default**: 1.0

#### chi_q

- **Type**: Real
- **Description**: 超参数 $\chi_q$，用于控制 $\tilde{q}(\mathbf{r})=\tanh(\chi_q q(\mathbf{r}))$ 的分布。
- **Default**: 1.0

### 3.7 核参数与非局域描述子

#### nkernel

- **Type**: Integer
- **Description**: 核函数个数，决定所有核相关数组（如 `kernel_type`、`gammanl`、`chi_xi`）长度。
- **Default**: 1

#### kernel_type

- **Type**: Integer 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定第 $i$ 个核函数的类型。
  - `1`: WT kernel（Wang-Teter）
  - `2`: Yukawa kernel（$k_{\rm{F}}^2\frac{\exp{({-\alpha k_{\rm{F}}|\mathbf{r}-\mathbf{r}'|})}}{|\mathbf{r}-\mathbf{r}'|}$），仅测试用。其中参数$\alpha$由`yukawa_alpha`设置
  - `3` 或 `4`: 从 `kernel_file` 读取离散核并插值，仅测试用
- **Default**: 全部为 1

#### kernel_scaling

- **Type**: Real 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定第 $i$ 个核函数的缩放参数 $\lambda$ 的倒数。缩放后的核函数为
  $$
  w_i(\mathbf{r}-\mathbf{r}')=\lambda^3 w_i'\!\left(\lambda(\mathbf{r}-\mathbf{r}')\right)
  $$
- **Default**: 全部为 1.0

#### yukawa_alpha

- **Type**: Real 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定第 $i$ 个核函数的参数 $\alpha$。仅在 `kernel_type = 2` 时使用。
- **Default**: 全部为 1.0

#### kernel_file

- **Type**: String 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定第 $i$ 个核函数的文件名。仅在 `kernel_type = 3` 或 `4` 时使用。
- **Default**: 全部为 `none`

#### gammanl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \gamma_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,\gamma(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### pnl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  p_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,p(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### qnl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  q_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,q(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### xi

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \xi(\mathbf{r})=\frac{\int w_i(\mathbf{r}-\mathbf{r}')\,\rho^{1/3}(\mathbf{r}')\,d\mathbf{r}'}{\rho^{1/3}(\mathbf{r})}
  $$
- **Default**: 全部为 `false`

#### tanhxi

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \tilde{\xi}(\mathbf{r})=\tanh\left(\chi_{\xi}\,\xi(\mathbf{r})\right)
  $$
- **Default**: 全部为 `false`

#### tanhxi_nl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \tilde{\xi}_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,\tilde{\xi}(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### tanh_pnl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \widetilde{p_{\mathrm{nl}}}(\mathbf{r})=\tanh\left(\chi_{p_{\mathrm{nl}}}\,p_{\mathrm{nl}}(\mathbf{r})\right)
  $$
- **Default**: 全部为 `false`

#### tanh_qnl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \widetilde{q_{\mathrm{nl}}}(\mathbf{r})=\tanh\left(\chi_{q_{\mathrm{nl}}}\,q_{\mathrm{nl}}(\mathbf{r})\right)
  $$
- **Default**: 全部为 `false`

#### tanhp_nl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \tilde{p}_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,\tilde{p}(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### tanhq_nl

- **Type**: Boolean 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于控制由第 $i$ 个核函数 $w_i(\mathbf{r}-\mathbf{r}')$ 定义的非局域描述子：
  $$
  \tilde{q}_{\mathrm{nl}}(\mathbf{r})=\int w_i(\mathbf{r}-\mathbf{r}')\,\tilde{q}(\mathbf{r}')\,d\mathbf{r}'
  $$
- **Default**: 全部为 `false`

#### chi_xi

- **Type**: Real 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定由第 $i$ 个核函数定义的非局域描述子对应的超参数 $\chi_{\xi}$：
  $$
  \tilde{\xi}(\mathbf{r})=\tanh\left(\chi_{\xi}\,\xi(\mathbf{r})\right)
  $$
- **Default**: 全部为 1.0

#### chi_pnl

- **Type**: Real 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定由第 $i$ 个核函数定义的非局域描述子对应的超参数 $\chi_{p_{\mathrm{nl}}}$：
  $$
  \widetilde{p_{\mathrm{nl}}}(\mathbf{r})=\tanh\left(\chi_{p_{\mathrm{nl}}}\,p_{\mathrm{nl}}(\mathbf{r})\right)
  $$
- **Default**: 全部为 1.0

#### chi_qnl

- **Type**: Real 列表
- **Description**: 包含 `nkernel`（见 `nkernel`）个元素。第 $i$ 个元素用于指定由第 $i$ 个核函数定义的非局域描述子对应的超参数 $\chi_{q_{\mathrm{nl}}}$：
  $$
  \widetilde{q_{\mathrm{nl}}}(\mathbf{r})=\tanh\left(\chi_{q_{\mathrm{nl}}}\,q_{\mathrm{nl}}(\mathbf{r})\right)
  $$
- **Default**: 全部为 1.0


## 4. 数据目录与文件要求

每个 `train_dir` / `validation_dir` 为一个构型目录。程序会按照训练类型、描述子设置加载需要的 `.npy` 文件。这些 `.npy` 通过ABACUS生成，下面简要介绍其类型与命名规则。

### 4.1 描述子相关文件

电子密度：

- `rho.npy`

半局域描述子：

- `gamma.npy`（当需要 `gamma` 或 `gammanl`）
- `p.npy`、`nablaRhox.npy`、`nablaRhoy.npy`、`nablaRhoz.npy`（当需要 `p`/`pnl`/`tanhp`相关）
- `q.npy`（当需要 `q`/`qnl`/`tanhq`相关）
- `tanhp.npy`（当需要 `tanhp` 或其非局域衍生）
- `tanhq.npy`（当需要 `tanhq` 或其非局域衍生）

非局域描述子

对于核索引 `ik`，程序按 `kernel_type[ik]` 和 `kernel_scaling[ik]`（$1/\lambda$，$\lambda$为缩放因子） 读取：

- `/<descriptor>_<kernel_type>_<kernel_scaling>.npy`

例如：

- `gammanl_1_1.npy`
- `pnl_2_0.5.npy`

对应描述子名可为：

- `gammanl`, `pnl`, `qnl`, `xi`, `tanhxi`, `tanhxi_nl`, `tanh_pnl`, `tanh_qnl`, `tanhp_nl`, `tanhq_nl`

### 4.2 标签文件（目标值）

- `energy_type = kedf`
  - Pauli能增强因子：`enhancement.npy`
  - 势标签（当 `loss=potential/both/both_new`）：`pauli.npy`
- `energy_type = exx`
  - 交换能增强因子：`enhancement_x.npy`
  - 势标签（当 `loss=potential/both/both_new`）：`v_pbe_x.npy`


## 5. `nnINPUT` 示例

下面是一个示例，用于训练使用5个核函数的$\mathrm{CPN}_5$动能泛函

```text
energy_type     kedf
nkernel         5
fftdim          27
nbatch          19683 # 27^3
ntrain          10
nvalidation     1

train_dir       /path/to/CD_Si    /path/to/AlP    /path/to/AlAs    /path/to/AlSb    /path/to/GaP    /path/to/GaAs    /path/to/GaSb    /path/to/InP    /path/to/InAs    /path/to/InSb
train_cell      fcc    fcc    fcc    fcc    fcc    fcc    fcc    fcc    fcc    fcc
train_a         10.334452882595281    10.395657712950090    10.817934591142382    11.721747701290822    10.313847974980126    10.765549669416206    11.644745211328591    11.205921368975233    11.647894717813344    12.441216584640049

validation_dir  /path/to/bcc_Al
validation_cell bcc
validation_a    6.125633010035168

loss            both
nepoch          60000
lr_start        0.005
lr_end          0.0001
lr_fre          5000
dump_fre        100
print_fre       5
feg_limit       3

device_type     gpu

kernel_scaling  2 1.5 1 0.75 0.5
chi_xi          0.6 0.8 1.0 1.5 3.0
chi_p           0.2 0.2 0.2 0.2 0.2
chi_q           0.1 0.1 0.1 0.1 0.1
chi_pnl         0.2 0.2 0.2 0.2 0.2
chi_qnl         0.1 0.1 0.1 0.1 0.1

nnode           100
nlayer          3

tanhxi          1 1 1 1 1
tanhxi_nl       1 1 1 1 1
tanhp           1
tanhp_nl        1 1 1 1 1
```

若用同样的描述子训练交换能，仅需将`energy_type`改成`exx`。

```text
energy_type     exx
```

此外，用户可以单独设定每个核函数的种类、参数、以及是否采用其描述子，比如采用如下设置也可以正常运行。

```text
nkernel         4
kernel_type     1 2 1 2
kernel_scaling  2 1.5 1 0.75

tanhxi          1 0 0 1
tanhxi_nl       0 1 1 1
tanhp           1
tanhp_nl        1 1 0 0
```

## 6.未来改进思路

- 目前程序默认训练集和验证集中的所有构型拥有相同的FFT维度，比如MPN和CPN泛函训练集中的构型都采用$27\times27\times27$的格点。未来可考虑放宽此限制。
- 目前程序默认每个构型FFT x, y, z三个维度的格点数相同，未来可考虑放宽此限制。
- 目前程序仅支持处理sc, bcc, fcc三种原胞，对于更复杂的原胞，需要拓展`Grid`类以生成对应倒空间格点。