from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Element

def ase_pmg_object_convert(model, convert_mode=0):
    '''
    在Pymatgen结构对象和ASE原子对象间进行转换。
    
    Args:
        model (Structure or Atoms): Pymatgen结构对象或ASE原子对象。
        convert_mode (int, optional): 转换模式:
            0: Structure转Atoms
            1: Atoms转Structure
            2: Atoms转Molecule
            默认为0。

    Returns:
        Structure or Atoms or Molecule: 转换后的对象。
    '''
    if convert_mode == 0:
        ase_obj = AseAtomsAdaptor.get_atoms(model)
        return ase_obj
    elif convert_mode == 1:
        pmg_stru = AseAtomsAdaptor.get_structure(model)
        return pmg_stru
    elif convert_mode == 2:
        pmg_mol = AseAtomsAdaptor.get_molecule(model)
        return pmg_mol
    else:
        raise ValueError('convert_mode should be 0, 1 or 2')

def input_convert(dict_info):
    """
    将键值全部转为大写或小写。
    
    Args:
        dict_info (dict): 输入的字典。
    
    Returns:
        dict: 转换后的字典。
    """
    new_dict = {}
    for i, j in dict_info.items():
        if i.islower():
            new_dict[i.upper()] = j
        elif i.isupper():
            new_dict[i.lower()] = j
        else:
            new_dict[i] = j
    return new_dict

def cal_oxygen_vacancy_formation_energy(bulk_dir=None, defect_dir=None, O2_E=-9.8543):
    """
    计算氧空位形成能。
    
    Args:
        bulk_dir (str, optional): 完整结构的计算目录。 Defaults to None.
        defect_dir (str, optional): 缺陷结构的计算目录。 Defaults to None.
        O2_E (float, optional): 氧气的能量。 Defaults to -9.8543.
    
    Returns:
        float: 氧空位形成能。
    """
    bulk_E = Vasprun(f'{bulk_dir}/vasprun.xml').final_energy
    defect_E = Vasprun(f'{defect_dir}/vasprun.xml').final_energy
    return defect_E + 0.5 * O2_E - bulk_E
    
def get_p_band_center(vasp_directory='dos', orbital=1, element='O'):
    """
    获取p轨道能带中心。
    
    Args:
        vasp_directory (str, optional): DOS计算的目录。 Defaults to 'dos'.
        orbital (int, optional): 轨道类型,s=0, p=1, d=2, f=3。 Defaults to 1.
        element (str, optional): 元素。 Defaults to 'O'.
    
    Returns:
        float: p轨道能带中心。
    """
    from pymatgen.electronic_structure.core import OrbitalType
    
    vasprun = Vasprun(f'./{vasp_directory}/vasprun.xml')
    dos_data = vasprun.complete_dos
    orb = OrbitalType(orbital)
    element = Element(element)
    p_band_center = dos_data.get_band_center(band=orb, elements=[element])
    return p_band_center