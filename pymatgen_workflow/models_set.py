import os
import glob
import itertools
import numpy as np
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
            
    @staticmethod
    def rotate_coordinate(filename='POSCAR',oz_string=None, ox_string=None):
        """
        旋转坐标系函数，用于读取 VASP 的 POSCAR 文件，定义新的坐标系并输出旋转后的结构文件。

        参数:
            filename (str): 输入的 POSCAR 文件名，默认为 'POSCAR'。
            oz_string (str): 定义新坐标系 z 轴方向的向量，支持输入一个向量（格式为 'x y z'）或两个点（格式为 'x1 y1 z1 x2 y2 z2'）。
            ox_string (str): 定义新坐标系 x 轴方向的向量，支持输入一个向量（格式为 'x y z'）或两个点（格式为 'x1 y1 z1 x2 y2 z2'）。

        功能:
            1. 解析 POSCAR 文件，提取晶格参数和原子坐标。
            2. 根据输入的 oz_string 和 ox_string 定义新的坐标系。
            3. 计算旋转矩阵，将原始坐标系旋转到新的坐标系。
            4. 输出旋转后的晶格和原子坐标到 'rotated.vasp' 文件。
            5. 生成包含超胞结构的 'rotated.xyz' 文件。

        输出:
            - 'rotated.vasp': 旋转后的晶格和原子坐标文件。
            - 'rotated.xyz': 包含超胞结构的 XYZ 文件。

        注意:
            - 输入的向量或点坐标应为分数坐标（fractional coordinates）。
            - 函数依赖 numpy 和 itertools 模块。

        示例:
            rotate_coordinate(filename='POSCAR', oz_string='0 0 1', ox_string='1 0 0')
        """
        vasp = open(filename, 'r').readlines()
        lattice = np.array([line.split() for line in vasp[2:5]], dtype='float')
        atom_type, atom_num = vasp[5].split(), np.array(vasp[6].split(), dtype='int')
        atoms = [t for t, n in zip(atom_type, atom_num) for _ in range(n)]
        natom = atom_num.sum()
        idx = 8 if vasp[7][0].lower() == 'd' else 9
        position = np.array([i.split() for i in vasp[idx:idx + natom]], dtype='float')
        position_xyz = np.dot(position, lattice)

        # **********************
        # * get final position *
        # **********************
        #
        # Step 1. get Vec(oz) normalize vector
        #
        # oz_string = input('Please enter one or two vector [fractional coordinates]\n'
        #                   '\tto define Vector of oz: \n')
        oz_string = oz_string
        temp = oz_string.split()
        if len(temp) == 3:
            oz = np.dot(np.array(temp, dtype='float'), lattice)
        elif len(temp) == 6:
            vec1 = np.array(temp[:3], dtype='float')
            vec2 = np.array(temp[3:], dtype='float')
            oz = np.dot(vec2 - vec1, lattice)
        oz /= np.linalg.norm(oz)
        #
        # Step 2. project Vec(ox) to plane(oxy) and normlize
        #         Vec_new(ox) = Vec(ox) - [Vec(ox) * Vec(oz)] * Vec[oz]
        #
        # ox_string = input('Please enter one or two vector [fractional coordinates]\n'
        #                   '\tto define Vector of ox: \n')
        ox_string=ox_string
        temp = ox_string.split()
        if len(temp) == 3:
            ox = np.dot(np.array(temp, dtype='float'), lattice)
        elif len(temp) == 6:
            vec1 = np.array(temp[:3], dtype='float')
            vec2 = np.array(temp[3:], dtype='float')
            ox = np.dot(vec2 - vec1, lattice)
        ox = ox - np.dot(ox, oz) * oz
        ox /= np.linalg.norm(ox)
        #
        # Step 3. cross production of oz, ox ot get oy
        #
        oy = np.cross(oz, ox)
        mat1 = np.concatenate((ox, oy, oz)).reshape((3, 3))
        rot = np.linalg.solve(mat1, np.identity(3))
        lattice_new = np.dot(lattice, rot)

        # *********************************************
        # * output to new position and check xyz file *
        # *********************************************
        # output to rotated.vasp
        vasp[2:5] = ['%20.12F %20.12F %20.12F\n' % tuple(line) for line in lattice_new]
        open('rotated.vasp', 'w').writelines(vasp)
        # supercell and output to rotated.xyz
        position_new = np.dot(position_xyz, rot)
        with open('rotated.xyz', 'w') as fb:
            fb.write(' %d\nmolecule\n' % (natom * 3**3))
            for i, j, k in itertools.product(*map(range, [3, 3, 3])):
                position_new2 = position_new + np.dot([i, j, k], lattice_new)
                for _type, (x, y, z) in zip(atoms, position_new2):
                    fb.write('%-2s %14.6F %14.6F %14.6F\n' % (_type, x, y, z))