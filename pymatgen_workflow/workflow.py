import os
from functools import wraps
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar  
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet, MVLNPTMDSet, MPMDSet
from pymatgen.analysis.eos import EOS


def deco_func(func):
    """
    装饰器函数,用于在执行被装饰的方法前检查计算是否已经完成。
    如果已完成,则直接返回;如果未完成,则执行原方法。
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        vasp_directory = kwargs.get('vasp_directory', func.__defaults__[0])
        if not os.path.exists(vasp_directory):
            return func(self, *args, **kwargs)
        else:
            self._MyWorkFlow__check_convergence(vasp_directory)
    return wrapper


class MyWorkFlow:
    def __init__(self, structure, workdir, incar_setting=None, k_setting=None, whether_run=True, vasp_version='std'):
        """
        MyWorkFlow类的构造函数。
        
        Args:
            structure (Structure): Pymatgen的Structure对象,表示要计算的结构。
            workdir (str): 工作目录。
            incar_setting (dict, optional): INCAR设置。 Defaults to None.
            k_setting (Union[dict, list, Kpoints], optional): KPOINTS设置。 Defaults to None.
            whether_run (bool, optional): 是否实际运行VASP。 Defaults to True.
            vasp_version (str, optional): VASP版本。 Defaults to 'std'.
        """
        self.structure = structure
        self.workdir = workdir + '_wf'
        self.init_workdir = os.getcwd()
        self.vasp_version = vasp_version
        self.incar_setting = incar_setting or {}
        self.k_setting = k_setting
        self.whether_run = whether_run

    @property
    def k_setting(self):
        """k_setting属性的getter方法。"""
        return self.__k_setting

    @k_setting.setter
    def k_setting(self, k_setting):
        """
        k_setting属性的setter方法。

        Args:
            k_setting (Union[dict, list, Kpoints]): KPOINTS设置。
        
        Raises:
            ValueError: 当k_setting不是dict,list或Kpoints对象时,抛出此异常。
        """
        if isinstance(k_setting, dict):
            self.__k_setting = k_setting
        elif isinstance(k_setting, list):
            self.__k_setting = Kpoints(kpts=[k_setting])
        elif isinstance(k_setting, Kpoints):
            self.__k_setting = k_setting
        else:
            raise ValueError("kpoints generation, k_setting should be a dict,a list or a Kpoints object")

    def show_kpts(self):
        """打印KPOINTS信息。"""
        if isinstance(self.k_setting, dict):
            kpts = Kpoints.automatic_density_by_vol(self.structure, self.k_setting["reciprocal_density"])
            print(kpts)
        elif isinstance(self.k_setting, Kpoints):
            print(self.k_setting)

    def start(self):
        """
        开始计算流程。
        创建工作目录,检查计算状态。
        """
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
            print(f'\033[0;32;1m----START------------------{self.workdir}-----------------------------------\033[0m')
            print(f'START:The work flow of\033[1;33;1m {self.workdir}\033[0m starts running!')
            os.chdir(self.workdir)
        else:
            print(f'\033[0;32;1m----START--------------------{self.workdir}-----------------------------------\033[0m')
            print(f'The model of\033[1;33;1m {self.workdir}\033[0m work flow file is already exited!')
            print(f'START:The work flow of {self.workdir} starts running!')
            os.chdir(self.workdir)
            self.check_all_job()

    def __check_convergence(self, job_name):
        """
        检查VASP计算是否收敛。
        
        Args:
            job_name (str): 任务名。
        
        Returns:
            bool: 如果收敛,返回True;否则返回False。
        """
        if os.path.exists(f'{job_name}/vasprun.xml'):
            v = Vasprun(f'{job_name}/vasprun.xml')
            return v.converged
        else:
            print(f'{job_name}/vasprun.xml is not existed!')
            return False

    def check_all_job(self):
        """检查所有VASP计算任务的状态。"""
        list_job_name = self.__auto_find_job()
        for job_name in list_job_name:
            if os.path.exists(f'{job_name}/vasprun.xml'):
                v = Vasprun(f'{job_name}/vasprun.xml')
                if v.converged:
                    print(f'{job_name} is converged!----Energy:{v.final_energy}')
                else:
                    print(f'{job_name} is not converged!')
            else:
                print(f'{job_name}/vasprun.xml is not existed!')

    def __auto_find_job(self):
        """自动查找所有的VASP任务。"""
        return [i for i in os.listdir() if os.path.isdir(i)]

    def __read_energy(self, vasp_directory):
        """
        读取VASP计算的能量。
        
        Args:
            vasp_directory (str): VASP计算目录。
        
        Returns:
            float: VASP计算得到的能量值。
        """
        return Vasprun(f'{vasp_directory}/vasprun.xml').final_energy

    def __is_continous(self, vasp_directory):
        """
        检查工作流是否连续。
        
        Args:
            vasp_directory (str): VASP计算目录。
        
        Raises:
            OSError: 如果计算未收敛,抛出此异常。
        """
        if not self.__check_convergence(vasp_directory):
            raise OSError('calculation is not converged!')
        else:
            print(f'{vasp_directory} finish!')
         
    @deco_func
    def relax(self, vasp_directory='relax', incar_set=None, k_set=None, potcar_set=None):
        """
        结构优化。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'relax'.
            incar_set (dict, optional): INCAR设置。 Defaults to None.
            k_set (Union[dict, list, Kpoints], optional): KPOINTS设置。 Defaults to None.
            potcar_set (tuple, optional): POTCAR设置,格式为(元素列表, 赝势类型)。 Defaults to None.
        """
        k_set = k_set or self.k_setting
        incar_set = incar_set or {}
        incar_set.update(self.incar_setting)
        
        relax = MPRelaxSet(self.structure, user_incar_settings=incar_set, user_kpoints_settings=k_set)
        relax.write_input(vasp_directory)
        
        if potcar_set:
            self.__set_potcar(symbol_list=potcar_set[0], functional=potcar_set[1], vasp_directory=vasp_directory)
        
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        self.structure = Structure.from_file('CONTCAR')
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @deco_func
    def opt(self, vasp_directory='opt', incar_set=None, k_set=None, potcar_set=None):
        """
        固定体积优化。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'opt'.
            incar_set (dict, optional): INCAR设置。 Defaults to None.
            k_set (Union[dict, list, Kpoints], optional): KPOINTS设置。 Defaults to None.
            potcar_set (tuple, optional): POTCAR设置,格式为(元素列表, 赝势类型)。 Defaults to None.
        """
        k_set = k_set or self.k_setting
        incar_set = incar_set or {}
        incar_set.update(self.incar_setting)
        incar_set.update({'ISIF': 2})
        
        opt = MPRelaxSet(self.structure, user_incar_settings=incar_set, user_kpoints_settings=k_set)
        opt.write_input(vasp_directory)
        
        if potcar_set:
            self.__set_potcar(symbol_list=potcar_set[0], functional=potcar_set[1], vasp_directory=vasp_directory)
        
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        self.structure = Structure.from_file('CONTCAR')
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @deco_func  
    def scf(self, vasp_directory='scf', incar_set=None, k_set=None, potcar_set=None):
        """
        自洽计算。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'scf'.
            incar_set (dict, optional): INCAR设置。 Defaults to None.
            k_set (Union[dict, list, Kpoints], optional): KPOINTS设置。 Defaults to None.
            potcar_set (tuple, optional): POTCAR设置,格式为(元素列表, 赝势类型)。 Defaults to None.
        """
        k_set = k_set or self.k_setting
        incar_set = incar_set or {}
        incar_set.update(self.incar_setting)
        
        scf = MPStaticSet(self.structure, user_incar_settings=incar_set, user_kpoints_settings=k_set)
        scf.write_input(vasp_directory)
        
        if potcar_set:
            self.__set_potcar(symbol_list=potcar_set[0], functional=potcar_set[1], vasp_directory=vasp_directory)
        
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @deco_func
    def band(self, vasp_directory='band', prev_calc_file='scf', density=20, dim=None):
        """
        能带结构计算。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'band'.
            prev_calc_file (str, optional): 前一次计算的目录。 Defaults to 'scf'.
            density (int, optional): kpoints密度。 Defaults to 20.
            dim (int, optional): 二维或三维结构的维数。 Defaults to None.
        """
        custom_settings = {
            'ISTART': 1,
            "ICHARG": 11,
            "LORBIT": 11,
            "LAECHG": "False",
            "LVHAR": "False"
        }
        band = MPNonSCFSet.from_prev_calc(prev_calc_file, mode="line", standardize=True, 
                                          user_incar_settings=custom_settings, kpoints_line_density=density)
        band.write_input(vasp_directory)
        
        os.system(f'cp {prev_calc_file}/CHGCAR {vasp_directory}')
        os.system(f'cp {prev_calc_file}/POSCAR {vasp_directory}')
        
        os.chdir(vasp_directory)
        if dim == 2:
            os.system('vaspkit -task 302')
            os.system('mv KPATH.in KPOINTS')
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @deco_func
    def dos(self, vasp_directory='dos', prev_calc_file='scf', density=100):
        """
        态密度计算。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'dos'.
            prev_calc_file (str, optional): 前一次计算的目录。 Defaults to 'scf'.
            density (int, optional): kpoints密度。 Defaults to 100.
        """
        custom_settings = {
            'ISTART': 1,
            "ICHARG": 11,
            "LORBIT": 11,
            "LAECHG": "False",
            "LVHAR": "False"
        }
        dos = MPNonSCFSet.from_prev_calc(prev_calc_file, mode="uniform", user_incar_settings=custom_settings, 
                                         reciprocal_density=density)
        dos.write_input(vasp_directory)
        
        os.system(f'cp {prev_calc_file}/CHGCAR {vasp_directory}')
        os.system(f'cp {prev_calc_file}/POSCAR {vasp_directory}')
        
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    def end(self):
        """结束计算流程,返回到初始目录。"""
        os.chdir(self.init_workdir)
        print(f'\033[0;31;1m END:The work flow of\033[1;33;1m {self.workdir}\033[0m \033[0;31;1mhas been completed! \033[0m')
        print(f'\033[0;31;1m----END------------------{self.workdir}-----------------------------------\033[0m')

    def MD(self,vasp_directory='md', ensemble='nvt', start_temp=None, end_temp=None, steps=100000, incar_set=None, k_set=None):
        """
        分子动力学模拟。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'md'.
            ensemble (str, optional): 系综类型,'nvt'或'npt'。 Defaults to 'nvt'.

            start_temp (float, optional): 初始温度。 Defaults to None.
            end_temp (float, optional): 结束温度。 Defaults to None.
            steps (int, optional): MD步数。 Defaults to 100000.
            incar_set (dict, optional): INCAR设置。 Defaults to None.
            k_set (Union[dict, list, Kpoints], optional): KPOINTS设置。 Defaults to None.
            """
        k_set = k_set or self.k_setting
        incar_set = incar_set or {}
        incar_set.update(self.incar_setting)
        if ensemble == 'npt':
            md = MVLNPTMDSet(self.structure, start_temp=start_temp, end_temp=end_temp, nsteps=int(steps), 
                            user_incar_settings=incar_set, user_kpoints_settings=k_set)
        elif ensemble == 'nvt':
            md = MPMDSet(self.structure, start_temp=start_temp, end_temp=end_temp, nsteps=int(steps), 
                        user_incar_settings=incar_set, user_kpoints_settings=k_set)
        
        md.write_input(vasp_directory)
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))

    def eos_fit(self, vasp_directory='eos', fitting_range=None):  
        """
        状态方程拟合。
        
        Args:
            vasp_directory (str, optional): 计算目录。 Defaults to 'eos'.
            fitting_range (list, optional): 拟合范围。 Defaults to None.
        """
        fitting_range = fitting_range or [0.6, 0.8, 1, 1.2, 1.4]
        Energies = []
        Lattice_volumes = []

        for i in fitting_range:
            self.structure.scale_lattice(self.structure.volume*i)
            Lattice_volumes.append(self.structure.volume)
            self.scf(vasp_directory=f'{vasp_directory}_{i}', incar_set={"LAECHG": "False", "LVHAR": "False", 
                                                                        "LWAVE": "False", "LCHARG": "False"})
            energy = self.__read_energy(vasp_directory=f'{vasp_directory}_{i}')
            Energies.append(energy)                       
            
        eos = EOS(eos_name='murnaghan')
        eos_fit = eos.fit(Lattice_volumes, Energies)
        plot_obj = eos_fit.plot(width=8, height=8, plt=plt, savefig=f'{vasp_directory}.png')
        plot_obj.show()
        plot_obj.savefig(f'{vasp_directory}.png')
        print(eos_fit.results)
    
    def __run_vasp(self, job=None):
        """
        运行VASP计算。
        
        Args:
            job (str, optional): 任务名。 Defaults to None.
        """
        if self.whether_run:
            os.system(f'mpirun vasp_{self.vasp_version} > log-{job}')
        else:
            print('Do not run vasp, only generate input files!')
        
    def __set_potcar(self, symbol_list, functional=None, vasp_directory=None):
        """
        设置POTCAR文件。
        
        Args:
            symbol_list (list): 元素列表。
            functional (str, optional): 赝势类型。 Defaults to None.
            vasp_directory (str, optional): VASP计算目录。 Defaults to None.
        """
        if vasp_directory:
            os.chdir(vasp_directory)
        potcar = Potcar()
        potcar.set_symbols(symbol_list, functional)
        potcar.write_file('POTCAR')
        os.chdir(os.path.dirname(os.getcwd()))