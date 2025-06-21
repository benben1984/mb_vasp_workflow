from pymatgen.core import Structure
from pymatgen_workflow import MyWorkFlow_PMG
from pymatgen_workflow.utils import build_models_list_and_folders_name

# 设置INCAR参数
low = {
    'ISMEAR': 0,
    'SIGMA': 0.05,
    'ISYM': 0,
    'ISPIN': 2,
    'EDIFF': 0.00001,
}

# 设置KPOINTS
K_l = [1, 1, 1]


# 从POSCAR文件构建结构对象和工作目录列表
structure, folder = build_models_list_and_folders_name('*POSCAR*')

# 遍历每个结构和工作目录
for stru, workdir in zip(structure, folder):
    # 创建MyWorkFlow_PMG对象
    mywork = MyWorkFlow_PMG(stru, workdir, incar_setting=low, k_setting=K_l)
  
    # 开始计算流程
    mywork.start()
    
    # 计算氧分子能量
    mywork.scf()

    
    # 结束计算流程
    mywork.end()