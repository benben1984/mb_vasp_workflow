# Pymatgen Workflow

Pymatgen Workflow 是一个基于 Pymatgen 的 VASP 计算工作流 Python 包。它提供了一个 `MyWorkFlow_PMG` 类,用于自动化各种 VASP 计算任务,如结构优化、自洽计算、能带结构和态密度计算等。

## 特点

- 自动化 VASP 计算流程
- 支持多种计算任务,如结构优化、自洽计算、能带结构和态密度计算等
- 方便计算氧空位形成能
- 易于扩展和定制

## 安装

1. 克隆此仓库:

```bash
git clone https://github.com/your_username/pymatgen_workflow.git

2.进入项目目录:

cd pymatgen_workflow

3.安装依赖项:

pip install -r requirements.txt

4.安装 pymatgen_workflow 包:

pip install -e .


使用示例
以下是一个使用 MyWorkFlow_PMG 类进行 VASP 计算的示例:
from pymatgen_workflow import MyWorkFlow_PMG
from pymatgen_workflow.utils import build_models_list_and_folders_name

# 设置INCAR参数
low = {
    'ALGO': 'Fast',
    'ENCUT': 550,
    'ISMEAR': 0,
    'LORBIT': 11,
    'ISPIN': 2,
    'EDIFF': 0.00001,
    'NPAR': 4,
    'PREC': 'Normal'
}

# 设置KPOINTS
K_l = [2, 2, 2]

# 从POSCAR文件构建结构对象和工作目录列表
structure, folder = build_models_list_and_folders_name('*POSCAR*')

# 遍历每个结构和工作目录
for stru, workdir in zip(structure, folder):
    # 创建MyWorkFlow_PMG对象
    mywork = MyWorkFlow_PMG(stru, workdir, incar_setting=low, k_setting=K_l)
    
    # 开始计算流程
    mywork.start()
    
    # 进行结构优化
    mywork.relax()
    
    # 进行自洽计算
    mywork.scf()
    
    # 计算能带结构
    mywork.band()
    
    # 计算态密度
    mywork.dos()
    
    # 计算氧空位形成能
    E = mywork.cal_oxygen_vacancy_formation_energy(
        bulk_dir='/path/to/bulk/scf',
        defect_dir='/path/to/defect/scf'
    )
    print(f'Oxygen vacancy formation energy: {E:.4f} eV')
    
    # 结束计算流程
    mywork.end()

更多使用示例请参见 examples 目录。

贡献
欢迎提出 Issue 和 Pull Request!如果您发现了任何 bug,或有任何改进建议,请随时与我们联系。

许可证
本项目采用 MIT 许可证。详情请参见 LICENSE 文件。