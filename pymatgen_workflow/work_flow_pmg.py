from work_flow_ase import MyWorkFlow
import os
import sys
import matplotlib.pyplot as plt
from functools import wraps
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet, MVLNPTMDSet, MPHSEBSSet, MVLElasticSet, MPMDSet
from pymatgen.analysis.eos import EOS,PolynomialEOS

class MyWorkFlow_PMG(MyWorkFlow):
    def __init__(self,
                 structure,
                 workdir,
                 incar_setting={},
                 k_setting=None,
                 whether_run=True,
                 vasp_version='std'):
        self.structure = structure
        self.workdir = workdir + '_wf'
        self.init_workdir = os.getcwd()
        self.vasp_version = vasp_version
        self.incar_setting = incar_setting
        self.k_setting = k_setting
        self.whether_run = whether_run

    @property
    def k_setting(self):

        return self.__k_setting

    @k_setting.setter
    def k_setting(self, k_setting):
        if isinstance(k_setting, dict):
            self.__k_setting = k_setting
        elif isinstance(k_setting, list):
            self.__k_setting = Kpoints(kpts=[k_setting])
        elif isinstance(k_setting, Kpoints):
            self.__k_setting = k_setting
        else:
            raise ValueError(
                "kpoints generation, k_setting should be a dict,a list or a Kpoints object")



#------------------------------------------run vasp--------------------------------------------

    def __run_vasp(self, job=None):
        if self.whether_run:
            os.system(f'mpirun vasp_{self.vasp_version} > log-{job}')
        else:
            print('Do not run vasp,only generate input files!')

#------------------------------------------pre_set vasp--------------------------------------------

    def show_kpts(self):

        if isinstance(self.k_setting, dict):
            kpts = Kpoints.automatic_density_by_vol(
                self.structure, self.k_setting["reciprocal_density"])
            print(kpts)
        elif isinstance(self.k_setting, Kpoints):
            print(self.k_setting)

    def __deco_func(func):
        para_tuple = func.__defaults__

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if kwargs == {}:
                vasp_directory = list(para_tuple)[0]
                if not os.path.exists(vasp_directory):
                    func(self, *args, **kwargs)
                else:
                    self.__check_convergence(vasp_directory)
            else:
                if 'vasp_directory' in kwargs:
                    vasp_directory = list(
                        para_tuple)[0] = kwargs['vasp_directory']
                else:
                    vasp_directory = list(para_tuple)[0]

                if not os.path.exists(vasp_directory):
                    func(self, *args, **kwargs)
                else:
                    self.__check_convergence(vasp_directory)
                    #os.chdir(os.path.dirname(os.getcwd()))

        return wrapper

    def __set_potcar(self, symbol_list, functional=None, vasp_directory=None):
        if vasp_directory != None:
            os.chdir(vasp_directory)
        potcar = Potcar()
        potcar.set_symbols(symbol_list, functional)
        potcar.write_file('POTCAR')
        os.chdir(os.path.dirname(os.getcwd()))

    def __set_kpts(self, k_input):
        if isinstance(k_input, dict):
            return k_input
        elif isinstance(k_input, list):
            return Kpoints(kpts=[k_input])
        elif isinstance(k_input, Kpoints):
            return k_input
        else:
            raise ValueError(
                "kpoints generation, k_setting should be a dict, a list or a Kpoints object")

    def __change_kpt_by_dim(self):
        s = self.structure
        a = s.lattice.a
        b = s.lattice.b
        c = s.lattice.c
        frac_coords = s.frac_coords
        x = frac_coords[:, 0]
        y = frac_coords[:, 1]
        z = frac_coords[:, 2]
        vac_a = (min(x) + 1 - max(x)) * a
        vac_b = (min(y) + 1 - max(y)) * b
        vac_c = (min(z) + 1 - max(z)) * c
        dim = [1 if x > 5 else 0 for x in [vac_a, vac_b, vac_c]]
        k_mesh = [int(x) for x in os.popen('sed -n 4p KPOINTS').read().split()]
        k_changed = " ".join(
            [str(k_mesh[x]) if dim[x] == 0 else '1' for x in range(3)])
        os.system("sed -i '4s/.*/%s/g' KPOINTS" % (k_changed))
        return dim

    def __set_incar(self, incar_setting={}):
        """
        set the INCAR file 
        """
        incar = Incar.from_file('INCAR')
        incar.update(incar_setting)
        incar.write_file('INCAR')
        pass

    def __auto_find_job(self):

        list_job_name = []
        for i in os.listdir():
            if os.path.isdir(i):
                list_job_name.append(i)
        return list_job_name

    def __read_energy(self, vasp_directory):
        
        return Vasprun(f'{vasp_directory}/vasprun.xml').final_energy

    
    def __check_convergence(self, job_name):
        "check vasp calculation convergence"
        if os.path.exists(f'{job_name}/vasprun.xml'):
            v = Vasprun(f'{job_name}/vasprun.xml')
            if v.converged:
                return True
            else:
                return False
        else:
            print(f'{job_name}/vasprun.xml is not existed!')

    def __is_continous(self, vasp_directory):
        """
        check the work_flow directory is continous or not
        """
        if self.__check_convergence(vasp_directory) == False:
            raise OSError('calculation is not converged!')
        else:
            print(f'{vasp_directory} finish!')
#------------------------------------------vasp process--------------------------------------------

    def start(self):
        """
        build the parent directory and enter the directory;
        check the work_flow directory;
        """
        import os
        #os.chdir(self.init_workdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
            print(
                f'\033[0;32;1m----START------------------{self.workdir}-----------------------------------\033[0m'
            )
            print(
                f'START:The work flow of\033[1;33;1m {self.workdir}\033[0m starts running!'
            )
            os.chdir(self.workdir)
        else:
            print(
                f'\033[0;32;1m----START--------------------{self.workdir}-----------------------------------\033[0m'
            )
            print(
                f'The model of\033[1;33;1m {self.workdir}\033[0m work flow file is already exited!'
            )
            print(f'START:The work flow of {self.workdir} starts running!')
            os.chdir(self.workdir)
            self.check_all_job()
        #os.chdir(self.workdir)

    def check_all_job(self):
        "check vasp calculation job by pymatgen"
        list_job_name = self.__auto_find_job()
        for job_name in list_job_name:
            if os.path.exists(f'{job_name}/vasprun.xml'):
                v = Vasprun(f'{job_name}/vasprun.xml')
                if v.converged:
                    print(
                        f'{job_name} is converged!----Energy:{v.final_energy}')
                else:
                    print(f'{job_name} is not converged!')
            else:
                print(f'{job_name}/vasprun.xml is not existed!')

    def error_continue(self, job_name, reset_para={}, reset_structure=True):
        """
        check the error of the calculation and continue the calculation
        """
        if self.__check_convergence(job_name) == False:
            os.chdir(job_name)
            if reset_structure:
                os.system('cp CONTCAR POSCAR')
            self.__set_incar(reset_para)
            '修改INCAR参数'
            # os.system('sed -i "$a ICHARG = 1" INCAR')
            # os.system('sed -i "$a ISTART = 1" INCAR')
            self.__run_vasp(job_name)
            os.chdir(os.path.dirname(os.getcwd()))
        else:
            pass
            os.chdir(os.path.dirname(os.getcwd()))

    @__deco_func
    def relax(self,
              vasp_directory='relax',
              OPTCELL=False,
              incar_set={},
              k_set=None,
              potcar_set=None):  ######## OPTCELL=[100 010 000]

        if k_set != None:
            k_set = self.__set_kpts(k_set)
        else:
            k_set = self.k_setting
        incar_set.update(self.incar_setting)
        relax = MPRelaxSet(self.structure,
                           user_incar_settings=incar_set,
                           user_kpoints_settings=k_set)
        relax.write_input(vasp_directory)
        if potcar_set != None:
            self.__set_potcar(symbol_list=potcar_set[0],
                              functional=potcar_set[1],
                              vasp_directory=vasp_directory)
        os.chdir(vasp_directory)
        dim = self.__change_kpt_by_dim()
        if OPTCELL:
            f = open('OPTCELL', 'w')
            for i in range(3):
                if dim[i] == 0:
                    oc = ("".join(['1' if x == i else '0' for x in range(3)]))
                else:
                    oc = '000'
                f.writelines(oc + '\n')
            f.close()
        self.__run_vasp(vasp_directory)
        self.structure = Structure.from_file('CONTCAR')
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @__deco_func
    def opt(self,
            vasp_directory='opt',
            incar_set={},
            k_set=None,
            potcar_set=None):

        if k_set != None:
            k_set = self.__set_kpts(k_set)
        else:
            k_set = self.k_setting
        incar_set.update(self.incar_setting)
        incar_set.update({'ISIF': 2})

        opt = MPRelaxSet(self.structure,
                         user_incar_settings=incar_set,
                         user_kpoints_settings=k_set)
        opt.write_input(vasp_directory)
        if potcar_set != None:
            self.__set_potcar(symbol_list=potcar_set[0],
                              functional=potcar_set[1],
                              vasp_directory=vasp_directory)
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        self.structure = Structure.from_file('CONTCAR')
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @__deco_func
    def scf(self,
            vasp_directory='scf',
            incar_set={},
            k_set=None,
            potcar_set=None):
        """_summary_

        Args:
            vasp_directory (str, optional): _description_. Defaults to 'scf'.
            incar_set (dict, optional): _description_. Defaults to {}.
            k_set (dict, optional): _description_. Defaults to {}.
            potcar_set (_type_, optional): a tuple like (a list of elements,'functional'). Defaults to None.
        """
        if k_set != None:
            k_set = self.__set_kpts(k_set)
        else:
            k_set = self.k_setting

        incar_set.update(self.incar_setting)

        scf = MPStaticSet(self.structure,
                          user_incar_settings=incar_set,
                          user_kpoints_settings=k_set)
        scf.write_input(vasp_directory)
        if potcar_set != None:
            self.__set_potcar(symbol_list=potcar_set[0],
                              functional=potcar_set[1],
                              vasp_directory=vasp_directory)
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    @__deco_func
    def band(self,
             vasp_directory='band',
             prev_calc_file='scf',
             density=20,
             dim=None):

        custom_settings = {
            'ISTART': 1,
            "ICHARG": 11,
            "LORBIT": 11,
            "LAECHG": "False",
            "LVHAR": "False"
        }
        band = MPNonSCFSet.from_prev_calc(prev_calc_file,
                                          mode="line",
                                          standardize=True,
                                          user_incar_settings=custom_settings,
                                          kpoints_line_density=density)
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

    @__deco_func
    def dos(self, vasp_directory='dos', prev_calc_file='scf', density=100):

        custom_settings = {
            'ISTART': 1,
            "ICHARG": 11,
            "LORBIT": 11,
            "LAECHG": "False",
            "LVHAR": "False"
        }
        dos = MPNonSCFSet.from_prev_calc(prev_calc_file,
                                         mode="uniform",
                                         user_incar_settings=custom_settings,
                                         reciprocal_density=density)
        dos.write_input(vasp_directory)
        os.system(f'cp {prev_calc_file}/CHGCAR {vasp_directory}')
        os.system(f'cp {prev_calc_file}/POSCAR {vasp_directory}')
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        self.__is_continous(vasp_directory)

    def end(self):
        """
        End the calculation and return to the parent directory
        """
        os.chdir(self.init_workdir)
        print(
            f'\033[0;31;1m END:The work flow of\033[1;33;1m {self.workdir}\033[0m \033[0;31;1mhas been completed! \033[0m'
        )
        print(
            f'\033[0;31;1m----END------------------{self.workdir}-----------------------------------\033[0m'
        )

    def MD(self,vasp_directory='md',ensemble='nvt',start_temp=None,end_temp=None, steps=100000, incar_set={}, k_set=None):
        
        if k_set != None:
            k_set = self.__set_kpts(k_set)
        else:
            k_set = self.k_setting
        incar_set.update(self.incar_setting)

        if ensemble=='npt':
            md=MVLNPTMDSet(self.structure,start_temp=start_temp,end_temp=end_temp,nsteps=int(steps),user_incar_settings=incar_set, user_kpoints_settings=k_set)
        elif ensemble == 'nvt':
            md=MPMDSet(self.structure,start_temp=start_temp,end_temp=end_temp,nsteps=int(steps),user_incar_settings=incar_set, user_kpoints_settings=k_set)
        md.write_input(vasp_directory)
        os.chdir(vasp_directory)
        self.__run_vasp(vasp_directory)
        os.chdir(os.path.dirname(os.getcwd()))
        #self.__is_continous(vasp_directory)


#-----------------------------------------combine functions--------------------------------------------
    def eos_fit(self,vasp_directory='eos',fitting_range=[0.6,0.8,1,1.2,1.4]):
        
        Energies = []
        Lattice_volumes = []

        for i in fitting_range:
            self.structure.scale_lattice(self.structure.volume*i)
            Lattice_volumes.append(self.structure.volume)
            self.scf(vasp_directory=f'{vasp_directory}_{i}',incar_set={"LAECHG": "False","LVHAR": "False","LWAVE": "False","LCHARG": "False"})
            energy=self.__read_energy(vasp_directory=f'{vasp_directory}_{i}')
            Energies.append(energy)                       
            
            
        eos = EOS(eos_name='murnaghan')
        eos_fit = eos.fit(Lattice_volumes, Energies)
        plot_obj=eos_fit.plot(width=8, height=8,plt=plt,savefig=f'{vasp_directory}.png')
        plot_obj.show()
        plot_obj.savefig(f'{vasp_directory}.png')
        print(eos_fit.results)
       
