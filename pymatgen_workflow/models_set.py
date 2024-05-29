import os
import glob
from ase.io import read
from ase.constraints import FixAtoms
from ase import Atom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.core import Structure
from pymatgen.core import Element
from copy import deepcopy

class ModelSet:
    """
    list all model setting for vasp calculations
    """
    @staticmethod
    def fix_atoms(model, fix_atom_Z_coordinate):
        """

        :param model: an model object
        :param fix_atom_Z_coordinate (Cartesian)
        :return: a constraint object
        note: need from ase.constraints import FixAtoms
        """
        n = model.get_positions()

        return FixAtoms(mask=n[:, 2] < fix_atom_Z_coordinate)
    
    @staticmethod
    def find_specified_atom_coordinates(model, atom_symbol):
        """

        :param model: a model object
        :param atom_symbol: the symbol of atom
        :return: a list of coordinates of the atom
        """
        from ase import Atoms
        from ase.io import read

        if isinstance(model, Atoms):
            atoms = model
        else:
            atoms = read(model)

        coordinates = []
        for i in atoms:
            if i.symbol == atom_symbol:
                coordinates.append(i.position)
        return coordinates

    @staticmethod
    def find_specified_atom_toppest_coordinates(model, atom_symbol):
        """

        :param model: a model object
        :param atom_symbol: the symbol of atom
        :return: x, y position of top atom
        """
        from ase import Atoms
        from ase.io import read

        if isinstance(model, Atoms):
            atoms = model
        else:
            atoms = read(model)

        coordinates = []
        for i in atoms:
            if i.symbol == atom_symbol:
                coordinates.append(i.position)
        coordinates.sort(key=lambda x:x[:][2],reverse=True)
        coordinate = coordinates[0][:2]
        return coordinate
    
    @staticmethod
    def set_magmoms(atoms,**kwargs):
        """

        :param atoms: a model object
        :param kwargs: the key is the symbol of atom, the value is the magmom of atom
        :return: a model object with magmoms
        """
        from ase import Atoms
        from ase.io import read

        if isinstance(atoms, Atoms):
            atoms = atoms
        else:
            atoms = read(atoms)

        magmoms = []
        for i in atoms.get_chemical_symbols():
            for key, value in kwargs.items():
                if i == key:
                    magmoms.append(value)
        atoms.set_initial_magnetic_moments(magmoms)
        return atoms
    
    @staticmethod
    def create_doping_eq_site(stru='POSCAR',element=None,doping_element=None):
        ran_stru = Structure.from_file(stru)
        O_sites_index = []
        for i in range(len(ran_stru.sites)):
            if ran_stru.sites[i].specie == Element(element):
                O_sites_index.append(i)
        spa = SpacegroupAnalyzer(ran_stru)
        spo = spa.get_space_group_operations()
        eqa = spa.get_symmetry_dataset()['equivalent_atoms']
        wkf = spa.get_symmetry_dataset()['wyckoffs']
        syms = SymmetrizedStructure(ran_stru,spo,eqa,wkf)
        eq_ids = syms.equivalent_indices
        list_index=[]
        for i in range(len(eq_ids)):
            for j in range(len(eq_ids[i])):
                if eq_ids[i][j] in O_sites_index:
                    list_index.append(i)
        list_index=list(set(list_index))
        ran_replace_stru=deepcopy(ran_stru)
        for n,i in enumerate(list_index):
            ran_replace_stru=deepcopy(ran_stru)
            ran_replace_stru.replace(i=eq_ids[i][0],species=doping_element)
            ran_replace_stru.to(filename=f'POSCAR-{n}',fmt="poscar")
    
    @staticmethod
    def create_vacancy_eq_site(stru='POSCAR',element='O'):
        """
        从给定的结构文件中创建具有指定元素空位的等价位点结构。
        
        参数:
        - stru (str): 结构文件的名称，默认为'POSCAR'。
        - element (str): 要创建空位的元素符号，默认为'O'。
        
        返回:
        无返回值，但会生成一系列含有空位的结构文件，命名为'POSCAR-0'，'POSCAR-1'，等。
        """
        # 从文件加载结构
        ran_stru = Structure.from_file(stru)
        
        # 筛选出指定元素的位点索引
        element_sites_index = [i for i in range(len(ran_stru.sites)) if ran_stru.sites[i].specie == Element(element)]

        # 分析结构的对称性
        spa = SpacegroupAnalyzer(ran_stru)
        spo = spa.get_space_group_operations()
        eqa = spa.get_symmetry_dataset()['equivalent_atoms']
        wkf = spa.get_symmetry_dataset()['wyckoffs']
        syms = SymmetrizedStructure(ran_stru,spo,eqa,wkf)
        
        # 获取等价原子索引
        eq_ids = syms.equivalent_indices
        
        # 根据等价原子索引和指定元素，扩展得到所有需要创建空位的索引
        list_index=[]
        for i in range(len(eq_ids)):
            list_index.extend(i for j in range(len(eq_ids[i])) if eq_ids[i][j] in element_sites_index)

        # 去除重复索引
        list_index=list(set(list_index))
        
        # 复制原始结构，依次为每个需创建空位的索引创建空位，并保存为文件
        ran_replace_stru=deepcopy(ran_stru)
        for n,i in enumerate(list_index):
            ran_replace_stru=deepcopy(ran_stru)
            ran_replace_stru.pop(eq_ids[i][0])
            ran_replace_stru.to(filename=f'POSCAR-{n}',fmt="poscar")