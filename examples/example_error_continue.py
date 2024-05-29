from pymatgen.core import Structure
from pymatgen_workflow import MyWorkFlow_PMG
from pymatgen_workflow.utils import build_models_list_and_folders_name
import logging
# 设置KPOINTS
K_l = [1, 1, 1]

structure,folder = build_models_list_and_folders_name('*POSCAR_O2*')
for stru, workdir in zip(structure,folder):
    mywork = MyWorkFlow_PMG(stru,workdir,k_setting=K_l)


    mywork.start()
    try:
        mywork.opt(incar_set={"NSW" :2})
    except Exception as opt_error:
        logging.warning(f"OPT计算错误信息：{opt_error}")
        mywork.error_continue("opt",reset_para={"NSW" :10}, reset_structure=True)
    try:
        mywork.scf(incar_set={"NELM" :5,"ISMEAR" : 0})
    except Exception as scf_error:
        logging.warning(f"SCF错误信息：{scf_error}")
        mywork.error_continue("scf",reset_para={"ICHARG" :1,"NELM":100},reset_structure=False)
    

    mywork.end()