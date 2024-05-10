from pymatgen.core import Structure
from pymatgen_workflow import MyWorkFlow_PMG

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
structure, folder = MyWorkFlow_PMG.build_models_list_and_folders_name('*POSCAR*')

# 遍历每个结构和工作目录
for stru, workdir in zip(structure, folder):
    # 创建MyWorkFlow_PMG对象
    mywork = MyWorkFlow_PMG(stru, workdir, incar_setting=low, k_setting=K_l)
    
    # 开始计算流程
    mywork.start()
    
    # 计算氧空位形成能
    E = mywork.cal_oxygen_vacancy_formation_energy(
        bulk_dir='/apps/users/lxy_maben/data/MnCr-pymatgen/MnCrMg-conventional/POSCAR_wf/scf',
        defect_dir='/apps/users/lxy_maben/data/MnCr-pymatgen/MnCr_O_vac_ATAT/MC_doped/POSCAR-25_wf/scf'
    )
    print(f'Oxygen vacancy formation energy: {E:.4f} eV')
    
    # 结束计算流程
    mywork.end()