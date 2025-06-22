from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Element
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar
import glob
import os
import numpy as np
from ase.io import read
def build_models_list_and_folders_name(*args):
    """

    :param args: different structure format.for examples:'*.cif','*CONTCAR*','*.xyz','*POSCAR*'
    :return: a tuple (models, folders_name) for calculation
    note: need import os, glob
    """
    models_f = []
    condition_list = args
    for i in condition_list:
        model = glob.iglob(i)
        for f in model:
            models_f.append(f)
    models_folders = [i for i in models_f if os.path.isfile(i)]
    models_folders_name = sorted(models_folders)
    models = [Structure.from_file(m) for m in models_folders_name]

    return models, models_folders_name

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

def parse_incar(input_data, from_file=True):
    """
    解析INCAR参数，可以从文件或字符串内容中读取
    
    Parameters:
    -----------
    input_data : str
        如果from_file为True，则为INCAR文件的路径；
        如果from_file为False，则为INCAR文件的字符串内容
    from_file : bool, optional
        指定是否从文件读取，默认为True
    
    Returns:
    --------
    dict
        包含VASP计算参数的字典
    
    Example:
    --------
    >>> parameters = parse_incar('INCAR', from_file=True)
    >>> print(parameters)
    {'ALGO': 'Fast', 'ENCUT': 550, 'ISMEAR': 0, 'LORBIT': 11, 'ISPIN': 2, 'EDIFF': 1e-05, 'NPAR': 4, 'PREC': 'Normal'}
    
    >>> incar_content = "ALGO = Fast\nENCUT = 550\nISMEAR = 0"
    >>> parameters = parse_incar(incar_content, from_file=False)
    >>> print(parameters)
    {'ALGO': 'Fast', 'ENCUT': 550, 'ISMEAR': 0}
    """
    try:
        if from_file:
            # 使用pymatgen的Incar类读取INCAR文件
            incar = Incar.from_file(input_data)
        else:
            # 使用pymatgen的Incar类从字符串读取
            incar_dict = {}
            for line in input_data.strip().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    incar_dict[key.strip()] = value.strip()
            incar = Incar.from_dict(incar_dict)
        
        # 将Incar对象转换为普通字典
        parameters = dict(incar)
        
        return parameters
        
    except FileNotFoundError:
        if from_file:
            print(f"错误: 找不到文件 {input_data}")
        return {}
    except Exception as e:
        print(f"解析INCAR时发生错误: {e}")
        return {}

def cal_oxygen_vacancy_formation_energy(bulk_dir=None, defect_dir=None, O2_E=-9.8543, bulk_coeff=1, defect_coeff=1):
    """
    计算氧空位形成能。
    
    Args:
        bulk_dir (str, optional): 完整结构的计算目录。其默认值为 None。
        defect_dir (str, optional): 缺陷结构的计算目录。其默认值为 None。
        O2_E (float, optional): 氧气分子的能量。默认值为 -9.8543。
        bulk_coeff (float, optional): 完整结构的能量系数。默认值为 1。
        defect_coeff (float, optional): 缺陷结构的能量系数。默认值为 1。
    
    Returns:
        float: 氧空位形成能，计算公式为 缺陷结构的最终能量 * 缺陷系数 + 氧气分子能量的一半 * 氧气系数 - 完整结构的最终能量 * 完整结构系数。
    """

    # 从完整结构的计算目录中读取最终能量
    bulk_E = Vasprun(f'{bulk_dir}/vasprun.xml').final_energy
    # 从缺陷结构的计算目录中读取最终能量
    defect_E = Vasprun(f'{defect_dir}/vasprun.xml').final_energy
    # 计算并返回氧空位形成能
    return defect_E * defect_coeff + 0.5 * O2_E - bulk_E * bulk_coeff
    
def get_p_band_center(vasp_directory='dos', orbital=1, element='O',erange= None):
    """
    获取p轨道能带中心。
    
    Args:
        vasp_directory (str, optional): DOS计算的目录。 Defaults to 'dos'.
        orbital (int, optional): 轨道类型,s=0, p=1, d=2, f=3。 Defaults to 1.
        element (str, optional): 元素。 Defaults to 'O'.
        erange (tuple, optional): 能量范围 (-10, 0)。 Defaults to None .
    
    Returns:
        float: p轨道能带中心。
    """
    from pymatgen.electronic_structure.core import OrbitalType
    
    vasprun = Vasprun(f'./{vasp_directory}/vasprun.xml')
    dos_data = vasprun.complete_dos
    orb = OrbitalType(orbital)
    element = Element(element)
    p_band_center = dos_data.get_band_center(band=orb, elements=[element],erange=erange)
    return p_band_center

def get_d_band_center(vasp_directory='dos', orbital=2, element=None,site_start=None,site_end=None, erange=None):
    """
    获取d轨道能带中心。
    
    Args:
        vasp_directory (str, optional): DOS计算的目录。 Defaults to 'dos'.
        orbital (int, optional): 轨道类型,s=0, p=1, d=2, f=3。 Defaults to 1.
        element (list, optional): 元素。 Defaults to 'None'.
        sites (list, optional): 站点。 Defaults to None.(cannot be used in conjunction with element)
        erange (tuple, optional): 能量范围 (-10, 0)。 Defaults to None .
    
    Returns:
        float: d轨道能带中心。
    """
    from pymatgen.electronic_structure.core import OrbitalType
    from pymatgen.core.sites import PeriodicSite
    vasprun = Vasprun(f'./{vasp_directory}/vasprun.xml')
    sites = vasprun.final_structure.sites
    select_sites = []
    for i in sites:
        if i.species_string == element:
            select_sites.append(i)
    dos_data = vasprun.complete_dos
    orb = OrbitalType(orbital)
    element = Element(element)
    sites = select_sites[site_start:site_end]
    if element != None and site_start == None:
        d_band_center = dos_data.get_band_center(band=orb, elements=[element],erange=erange)
    elif element != None and site_start != None:
        d_band_center = dos_data.get_band_center(band=orb ,sites=sites,erange=erange)

    return d_band_center

def get_average_metal_oxygen_bond_length(file_name='POSCAR', oxygen_symbol='O', metal_symbols=[], cutoff=2.5):
    """Calculate the overall average bond length between all different metals and oxygen.

    Args:
        file_name (str, optional): Name of the file to read the structure from. Defaults to 'POSCAR'.
        oxygen_symbol (str, optional): Symbol of the oxygen atom. Defaults to 'O'.
        metal_symbols (list, optional): List of symbols for the metal atoms. Defaults to ['Mn', 'Cr', 'Mg'].
        cutoff (float, optional): Maximum distance for considering a bond. Defaults to 2.5.

    Returns:
        float: The overall average bond length between all different metals and oxygen.
    """
    # Read the POSCAR file
    structure = read(file_name, format='vasp')

    # Extract coordinates of oxygen and metal atoms from the structure
    oxygen_coords = []
    metal_coords = []

    for atom in structure:
        symbol = atom.symbol
        coords = atom.position
        if symbol == oxygen_symbol:
            oxygen_coords.append(coords)
        elif symbol in metal_symbols:
            metal_coords.append(coords)

    # Calculate the bond lengths between each metal and oxygen
    bond_lengths = []
    for metal_coord in metal_coords:
        for oxygen_coord in oxygen_coords:
            if np.linalg.norm(metal_coord - oxygen_coord) < cutoff:
                bond_lengths.append(np.linalg.norm(metal_coord - oxygen_coord))

    # Calculate the overall average bond length
    overall_average_bond_length = sum(bond_lengths) / len(bond_lengths)

    return overall_average_bond_length

def cal_surface_energies(slab_models, bulk_models, E_relax_slab, E_unrelax_slab, E_bulk):
        """

        :param slab_models: a list of slab models
        :param bulk_models: a list of bulk models
        :param E_relax_slab: a list of energies of relaxation slab
        :param E_unrelax_slab: a list of energies of unrelaxation slab
        :param E_bulk: a list of energies of bulks
        :return: a list of surface energies
        """
        # surface areas
        areas = []
        for slab in slab_models:
            cell = slab.get_cell()
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            areas.append(area)
            # n*bulk_atomic_number=slab_number
            # n = [slab.get_global_number_of_atoms() / bulk.get_global_number_of_atoms() for (slab,bulk) in zip(slab_models,bulk_models)]
        s_n = [slab.get_global_number_of_atoms() for slab in slab_models]
        b_n = [bulk.get_global_number_of_atoms() for bulk in bulk_models]
        n = np.array(s_n) / np.array(b_n)
        # calculation for surface energies
        E_relax_slab = np.array(E_relax_slab)
        E_unrelax_slab = np.array(E_unrelax_slab)
        E_bulk = np.array(E_bulk)
        # n = np.array(n)
        areas = np.array(areas)
        surf_E = (E_unrelax_slab - n * E_bulk) / (2 * areas) + (E_relax_slab - E_unrelax_slab) / areas
        return surf_E