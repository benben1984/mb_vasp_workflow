import os
from re import T
import sys
import numpy as np
from functools import wraps
from ase.io import read
from ase.calculators.vasp import Vasp
from ase import Atoms
from ase.io.vasp import read_vasp_out
from ase.eos import EquationOfState
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt


class MyWorkFlow():
    def __init__(self, atoms, workdir,common_parameters={},cal='vasp'):
        """
        atoms:ase.Atoms object
        workdir:the directory of the model
        common_parameters:common parameters of vasp calculator,such as NCORE,U value;for example:{'xc':'PBE'}
        """
        self.atoms = atoms
        self.workdir = workdir+'_wf'
        self.common_parameters = common_parameters
        self.calc = cal            #Vasp(**common_parameters)
        self.init_workdir = os.getcwd()
        self.band_gap = None
        self.vbm = None
        self.cbm = None
        self.is_band_gap_direct = None

    @property
    def cal(self):
        return self.__calc

    @cal.setter
    def cal(self,cal):
        if cal == 'vasp':
            self.__calc = Vasp(**self.common_parameters)
        else:
            self.__calc = None
            
    def start(self):
        """
        build the parent directory and enter the directory;
        check the work_flow directory;
        """
        import os
        #os.chdir(self.init_workdir)
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
            print(f'\033[0;32;1m----START------------------{self.workdir}-----------------------------------\033[0m')
            print(f'START:The work flow of\033[1;33;1m {self.workdir}\033[0m starts running!')
        else:
            print(f'\033[0;32;1m----START--------------------{self.workdir}-----------------------------------\033[0m')
            print(f'The model of\033[1;33;1m {self.workdir}\033[0m work flow file is already exited!')
            print(f'START:The work flow of {self.workdir} starts running!')
        os.chdir(self.workdir)
        
    def check_convergence(self,vasp_directory = None):
        """
        vas_directory:the directory that assigned to check convergence
        """
        
        os.chdir(self.init_workdir)
        if os.path.exists(self.workdir):
            os.chdir(self.workdir)
            if vasp_directory == None:
                for i in os.listdir():
                    if os.path.isdir(i):
                        os.chdir(i)
                        if self.calc.read_convergence():
                            print(f'The model in {self.workdir}_{i} is converged!')
                        else:
                            print(f'The model in\033[0;33;1m {self.workdir}_{i}\033[0m is\033[0;33;1m not converged! \033[0m')

                        os.chdir(os.path.dirname(os.getcwd()))
            else:
                os.chdir(vasp_directory)
                if self.calc.read_convergence():
                    print(f'The model in {self.workdir}_{vasp_directory} is converged!')
                else:
                    print(f'The model in\033[0;33;1m {self.workdir}_{vasp_directory}\033[0m is\033[0;33;1m not converged! \033[0m')
                os.chdir(os.path.dirname(os.getcwd()))
        else:
            print(f'The model of\033[0;33;1m {self.workdir}\033[0m is\033[0;33;1m not exited! \033[0m')

    def __deco_func(func):
        para_tuple=func.__defaults__
        @wraps(func)
        def wrapper(self,*args,**kwargs):
            if kwargs == {}:
                vasp_directory = list(para_tuple)[0]
                if not os.path.exists(vasp_directory):
                    func(self,*args,**kwargs)
                else:
                    print(f'The model of {self.workdir}_{vasp_directory} is already exited!')
                    os.chdir(vasp_directory)
                    if self.calc.read_convergence():
                        print(f'The model in {self.workdir}_{vasp_directory} is converged!')
                    else:
                        print(f'The model in {self.workdir}_{vasp_directory} is not converged!')
                    os.chdir(os.path.dirname(os.getcwd()))
            else:
                if 'vasp_directory' in kwargs:
                    vasp_directory=list(para_tuple)[0]=kwargs['vasp_directory']
                else:
                    vasp_directory = list(para_tuple)[0]

                if not os.path.exists(vasp_directory):
                    func(self,*args,**kwargs)
                else:
                    print(f'The model of {self.workdir}_{vasp_directory} is already exited!')
                    os.chdir(vasp_directory)
                    if self.calc.read_convergence():
                        print(f'The model in {self.workdir}_{vasp_directory} is converged!')
                    else:
                        print(f'The model in {self.workdir}_{vasp_directory} is not converged!')
                    os.chdir(os.path.dirname(os.getcwd()))
        return wrapper

    @__deco_func
    def relax(self,vasp_directory = 'relax',common_para=True,ibrion=2,ediffg=-0.02,nsw=200,potim=0.2,**kwargs):
        """
        vasp_directory:the directory of vasp calculation
        common_para:whether use the common parameters
        ibrion:the ibrion default parameter of vasp
        ediffg:the ediffg default parameter of vasp
        nsw:the nsw default parameter of vasp
        potim:the potim default parameter of vasp
        kwargs:the other parameters of vasp
        """
        os.makedirs(vasp_directory)
        os.chdir(vasp_directory)
        if common_para:
            kwargs.update(self.common_parameters,ibrion=ibrion,ediffg=ediffg,nsw=nsw,potim=potim,**kwargs)
            self.calc = Vasp(**kwargs)
        else:
            self.calc = Vasp(ibrion=ibrion,ediffg=ediffg,nsw=nsw,potim=potim,**kwargs)
        self.atoms.set_calculator(self.calc)
        self.atoms.get_potential_energy()
        self.atoms = read('CONTCAR')        
        os.chdir(os.path.dirname(os.getcwd()))
        return self.atoms

    def opt(self,vasp_directory = 'opt',common_para=True,lattice_scale=[0.96,0.98,1,1.02,1.04],**kwargs):
        """
        opt:optimize the lattice parameters by Energy of States
        vasp_directory:the directory of vasp calculation
        common_para:whether use the common parameters
        kwargs:the other parameters of vasp
        """
        Energies = []
        Lattice_volumes = []
        count=0
        if not os.path.exists(vasp_directory):
            os.makedirs(vasp_directory)
            os.chdir(vasp_directory)

            for i in lattice_scale:
                self.atoms.set_cell(i*self.atoms.get_cell(), scale_atoms=True)
                count+=1
                vasp_directory ='{}-opt'.format(lattice_scale[count-1])
                self.scf(vasp_directory = vasp_directory,common_para=common_para,**kwargs)
                Energies.append(self.get_energy(vasp_directory=vasp_directory))
                Lattice_volumes.append(self.atoms.get_volume())
        #######plot the Energy of States ############3
            import matplotlib.pyplot as plt
            eos = EquationOfState(Lattice_volumes, Energies)
            v0, e0, B = eos.fit()
            eos.plot('EOS_fitting.png',show=False)
            plt.plot(Lattice_volumes,Energies)
            plt.xlabel('Lattice_volumes')
            plt.ylabel('Energy')
            plt.savefig('Energies_vs_Volumes.png')
        else:
            print(f'The model of {self.workdir}_{vasp_directory} is already exited!')
            for i in lattice_scale:
                count+=1
                vasp_directory ='{}-opt'.format(lattice_scale[count-1])
                self.check_convergence(vasp_directory=vasp_directory)
        return Energies,Lattice_volumes

    @__deco_func
    def scf(self,vasp_directory = 'scf',common_para=True,**kwargs):
        """
        Run a single-point calculation with ASE VASP calculator.
        parameters:input parameters of Vasp calculator;for example:xc='PBE'
        """        
        os.makedirs(vasp_directory)
        os.chdir(vasp_directory)
        if common_para:
            kwargs.update(self.common_parameters,**kwargs)
            self.calc = Vasp(**kwargs)
        else:
            self.calc = Vasp(**kwargs)
        self.atoms.set_calculator(self.calc)
        self.atoms.get_potential_energy()
        os.chdir(os.path.dirname(os.getcwd()))

    @__deco_func
    def dos(self,vasp_directory = 'dos',common_para=True,istart=1,icharg=11,lorbit=11,nedos=1100,**kwargs):
        """
        Run a dos calculation with ASE VASP calculator.
        parameters:input parameters of Vasp calculator;for example:xc='PBE'
        Note:when k number > 4, ISMEAR=-5 is recommended
        """
        os.makedirs(vasp_directory)
        os.system(f'cp ./scf/CHGCAR ./{vasp_directory}/')
        os.chdir(vasp_directory)
        if common_para:
            kwargs.update(self.common_parameters,istart=istart,icharg=icharg,lorbit=lorbit,nedos=nedos,**kwargs)
            self.calc = Vasp(**kwargs)
        else:
            self.calc = Vasp(istart=istart,icharg=icharg,lorbit=lorbit,nedos=nedos,**kwargs)
        self.atoms.set_calculator(self.calc)
        self.atoms.get_potential_energy()
        os.chdir(os.path.dirname(os.getcwd()))
       
    @__deco_func
    def band(self,vasp_directory = 'band',common_para=True,band_path=None,npoints=10,special_points=None,density=None,cofficient=1,istart=1,icharg=11,lorbit=11,nedos=1100,**kwargs):
        """
        Run a band structure calculation with ASE VASP calculator.
        parameters:input parameters of Vasp calculator;for example:xc='PBE',Note:when k number > 4, ISMEAR=-5 is recommended;
        band_path:the path of band structure
        npoints:number of k-points along the path
        special_points:the special points of band structure
        density:the density of k-points along the path
        cofficient:the cofficient for increasing NBANDS;for example:cofficient=1.3 means NBANDS=1.3*NBANDS 
        """
        from ase.dft.kpoints import bandpath
        
        os.makedirs(vasp_directory)
        os.system(f'cp ./scf/CHGCAR ./{vasp_directory}/')
        #读取scf的NBAND值，设置kpoints的line-mode
        with open('./scf/OUTCAR','r') as f:
            for line in f.readlines():
                if 'NBANDS' in line:
                    nbands = int(line.split()[-1])
                    break
        nbands = round(nbands*cofficient) #设置NBANDS的扩大系数，用于计算精准能带结构
        path = self.atoms.cell.bandpath(path=band_path,npoints=npoints,special_points=special_points,density=density)
        #path = bandpath(band_path, self.atoms.get_cell(), npoints=npoints)       
        os.chdir(vasp_directory)
        if common_para:
            kwargs.update(self.common_parameters,istart=istart,icharg=icharg,lorbit=lorbit,nedos=nedos,nbands=nbands,reciprocal=True,kpts=path.kpts,**kwargs)
            self.calc = Vasp(**kwargs)
        else:
            self.calc = Vasp(istart=istart,icharg=icharg,lorbit=lorbit,nedos=nedos,nbands=nbands,reciprocal=True,kpts=path.kpts,**kwargs)
        self.atoms.set_calculator(self.calc)
        self.atoms.get_potential_energy()
        bs = self.calc.band_structure() # ASE Band structure object
        bs.plot(show=True,filename='bandstru')    # Plot the band structure
        os.chdir(os.path.dirname(os.getcwd()))
        
    def phonon(self,vasp_directory = 'phonon',common_para=True,band_path=None,npoints=10,**kwargs):
        pass

    def end(self):
        """
        End the calculation and return to the parent directory
        """
        os.chdir(self.init_workdir)
        print(f'\033[0;31;1m END:The work flow of\033[1;33;1m {self.workdir}\033[0m \033[0;31;1mhas been completed! \033[0m')
        print(f'\033[0;31;1m----END------------------{self.workdir}-----------------------------------\033[0m')
    
    def check_job(self):
        """
        check the vasp work flow job status
        """
        os.chdir(self.init_workdir)
        if os.path.exists(self.workdir):
            os.chdir(self.workdir)
            for i in os.listdir():
                if os.path.isdir(i):
                    os.chdir(i)
                    if os.path.exists('OUTCAR'):
                        with open('OUTCAR','r') as f:
                            for line in f.readlines():
                                if 'Voluntary context switches' in line:
                                    if int(line.split()[-1]) == 0:
                                        print(f'{self.workdir} {i} is running')
                                    else:
                                        print(f'{self.workdir} {i} is finished')
                    os.chdir(os.path.dirname(os.getcwd()))
        else:
            print(f'{self.workdir} is not exist')

    def check_input(self):
        """
        get the vasp input files
        """
        print(self.calc.asdict()['inputs'])



        
##################pre-process########################
    @staticmethod
    def build_models_list_and_folders_name(*args):
        """

        :param args: different structure format.for examples:'*.cif','*CONTCAR*','*.xyz','*POSCAR*'
        :return: a tuple (models, folders_name) for calculation
        note: need import os, glob, from ase.io import read
        """
        import os
        import glob
        from ase.io import read

        models_f = []
        condition_list = args
        for i in condition_list:
            model = glob.iglob(i)
            for f in model:
                models_f.append(f)
        models_folders = [i for i in models_f if os.path.isfile(i)]
        models_folders_name = sorted(models_folders)
        models = [read(m) for m in models_folders_name]
        return models, models_folders_name

    def set_magmoms(self,**kwargs):
        """

        :param atoms: a model object
        :param kwargs: the key is the symbol of atom, the value is the magmom of atom; for example:CuFe2O4,set_magmoms(Fe=5.0,Cu=5.0,O=1.0)
        :return: a model object with magmoms
        """

        magmoms = []
        for i in self.atoms.get_chemical_symbols():
            for key, value in kwargs.items():
                if i == key:
                    magmoms.append(value)
        self.atoms.set_initial_magnetic_moments(magmoms)
        return self.atoms

    def fix_slab_atoms(self, fix_atom_Z_coordinate=None):
        """

        :model: an Atoms object
        :param fix_atom_Z_coordinate (Cartesian)
        :note: c is a constraint object
        note: need from ase.constraints import FixAtoms
        """
        n = self.atoms.get_positions()
        c = FixAtoms(mask=n[:, 2] < fix_atom_Z_coordinate)
        self.atoms.set_constraint(c)
        


##################post-processing####################  
    def plot_dos(self,vasp_directory='dos',projector_spd=False,projector_element=False):
        """
        parameters:vasp_directory:the directory of vasp calculation
        project_spd:if True,project the dos on the spd orbital 
        projector_element:if True,project the dos on every element
        """
        
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.electronic_structure import plotter
        from pymatgen.electronic_structure.plotter import DosPlotter
               
        vasprun = Vasprun('./{}/vasprun.xml'.format(vasp_directory))
        dos_data = vasprun.complete_dos

        if projector_spd:
            plotter = DosPlotter(stack=False)
            plotter.add_dos("Total DOS", dos=dos_data)
            plotter.add_dos_dict(dos_data.get_spd_dos())
            plotter.save_plot('{}-plt_projector_spd_dos.png'.format(vasp_directory),'png')
            
        elif projector_element:
            plotter = DosPlotter(stack=False)
            plotter.add_dos("Total DOS", dos=dos_data)
            plotter.add_dos_dict(dos_data.get_element_dos())
            plotter.save_plot('{}-plt_projector_elements_dos.png'.format(vasp_directory),'png')
            
        else:
            plotter = DosPlotter(stack=False)
            plotter.add_dos("Total DOS", dos=dos_data)
            plotter.save_plot('{}-plt_total_dos.png'.format(vasp_directory),'png')

    def plot_band(self,vasp_directory='band',band_plus_dos=False):
        """
        vasp_directory:the directory of vasp calculation
        band_plus_dos:if True,plot the band and dos together
        """
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.electronic_structure import plotter
        from pymatgen.electronic_structure.plotter import BSPlotter,BSDOSPlotter
        from matplotlib import pyplot as plt
        vasprun = Vasprun('./{}/vasprun.xml'.format(vasp_directory))
        band_data = vasprun.get_band_structure(line_mode=True)
        if band_plus_dos:
            dos_data = vasprun.complete_dos
            plotter = BSDOSPlotter(bs_projection=None,dos_projection=None)
            plotter.get_plot(bs=band_data, dos=dos_data)
            plt.savefig('{}-plt_band_plus_dos.png'.format(vasp_directory))
        else:
            plotter = BSPlotter(band_data)
            plotter.get_plot()
            plotter.save_plot('{}-plt_band.png'.format(vasp_directory),'png')

    def check_time(self,vasp_directory='scf'):
        """
        vasp_directory:the directory of vasp calculation
        return the calculation time (sec)
        """
        from pymatgen.io.vasp.outputs import Outcar
        vasp_outcar = Outcar('./{}/OUTCAR'.format(vasp_directory))
        print('The calculation stats are :{} '.format(vasp_outcar.run_stats))
        print('The calculation time is {} s'.format(vasp_outcar.run_stats['Total CPU time used (sec)']))
        return vasp_outcar.run_stats['Total CPU time used (sec)']

    def get_energy(self,vasp_directory='scf'):
        """
        vasp_directory:the directory of vasp calculation
        """
        os.chdir(self.init_workdir)
        if os.path.exists(self.workdir):
            os.chdir(self.workdir)
            if os.path.exists(vasp_directory):
                os.chdir(vasp_directory)
                if os.path.exists('OUTCAR'):
                    energy = read_vasp_out('OUTCAR').get_potential_energy()
                    os.chdir(os.path.dirname(os.getcwd()))
                    return energy
                else:
                    print(f'The OUTCAR in {vasp_directory} is not exist')
            else:
                print(f'{self.workdir} {vasp_directory} is not exist')
        else:
            print(f'{self.workdir} is not exist')
        
    def get_band_gap_info(self,vasp_directory='band'):
        "采用pymatgen库获取vasp计算能带带隙"
        from pymatgen.io.vasp.outputs import Vasprun
        os.chdir(self.init_workdir)
        if os.path.exists(self.workdir):
            os.chdir(self.workdir)
            if os.path.exists(vasp_directory):
                os.chdir(vasp_directory)
                if os.path.exists('vasprun.xml'):
                    vasprun = Vasprun('vasprun.xml')
                    self.band_gap, self.cbm, self.vbm, self.is_band_gap_direct = vasprun.eigenvalue_band_properties
                    os.chdir(os.path.dirname(os.getcwd()))
                    return self.band_gap, self.cbm, self.vbm, self.is_band_gap_direct
                else:
                    print(f'The vasprun.xml in band is not exist')
            else:
                print(f'{self.workdir} band is not exist')
        else:
            print(f'{self.workdir} is not exist')



##############################combine the above functions######################
    def test_convergence(self,para_list=[], para_name=None):
        """
        This function is used to test the convergence of a given input parameter
        para_list:the list of the input parameter that needs to test its convergence
        para_name:the name of the input parameter
        return: a tuple of the energy and the cost time

        """

        Energies = []
        Cost_time = []
        for para in para_list:
            test_para = {para_name:para}
            self.common_parameters.update(**test_para)
            self.scf(vasp_directory=f'{para_name}_{para}')
            energy = self.get_energy(vasp_directory=f'{para_name}_{para}')
            time = self.check_time(vasp_directory=f'{para_name}_{para}')
            Energies.append(energy)
            Cost_time.append(time)
            print(f'{para_name} = {para}, energy = {energy} eV')
        return Energies, Cost_time

    def plot_test_convergence(self,para_list=[], para_name=None, Energies=[], Cost_time=[]):

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(para_list, Energies, 'bo-')
        ax1.set_xlabel(f'{para_name}')
        ax1.set_ylabel('Energy (eV)', color='b')
        ax2= ax1.twinx()
        ax2.plot(para_list, Cost_time, 'ro-')
        ax2.set_ylabel('Cost time (s)', color='r')
        plt.savefig(f'{self.workdir}-{para_name}_convergence.png')

    def cal_bader(self,vasp_directory='bader'):

        from ase.io.bader import attach_charges
        self.scf(vasp_directory=vasp_directory,prec='Accurate',laechg=True)        
        os.chdir(vasp_directory)
        if 'ACF.dat' not in os.listdir("."):
            os.system('chgsum.pl AECCAR0 AECCAR2')
            os.system('bader CHGCAR -ref CHGCAR_sum')
        else:
            print('ACF.dat exists,reading it>>>>>>>>>>>>>>>>>>')
        attach_charges(self.atoms, 'ACF.dat')
        os.chdir(os.path.dirname(os.getcwd()))
        for atom in self.atoms:
            print('Atom', atom.symbol, 'Bader charge', atom.charge)

    def cal_work_function(self,vasp_directory='work_function'):

        from pymatgen.analysis.surface_analysis import WorkFunctionAnalyzer
        self.scf(vasp_directory=vasp_directory,prec='Accurate',lvhar=True,isym=0,idipol=3,ldipol=True)
        os.chdir(vasp_directory)
        if 'LOCPOT' and 'OUTCAR'in os.listdir(".") and self.calc.read_convergence():
            wfa = WorkFunctionAnalyzer.from_files('POSCAR', 'LOCPOT', 'OUTCAR')
            work_function = wfa.work_function
            print('The work function is {} eV'.format(work_function))
            wfa.get_labels(plt)
            wfa.get_locpot_along_slab_plot()
            return work_function
        else:
            print('Need to restart the calculation of work function')
        os.chdir(os.path.dirname(os.getcwd()))

    def cal_elf(self,vasp_directory='elf'):
        """
        This function is used to calculate the electron localization function
        """
        if 'npar' or 'ncore' in self.common_parameters.keys():
            print('The calculation of ELF is not supported in parallel mode,please remove npar or ncore from INCAR!')
        else:
            self.scf(vasp_directory=vasp_directory,prec='Accurate',lelf=True)

    def cal_freq(self,vasp_directory='freq',fix_atom_Z_coordinate=None):

        self.fix_slab_atoms(fix_atom_Z_coordinate=fix_atom_Z_coordinate)
        self.relax(vasp_directory=vasp_directory,ibrion=5,potim=0.015,nsw=1,nfree=2,ediff=1e-7)

    @staticmethod
    def cal_neb(initial,final,image_number=None,idpp=True,vasp_directory='neb',**kwargs):
        """
        This function is used to calculate the NEB;
        intial_state:the initial state of the NEB calculation;it should be a string of the structure file name
        final_state:the final state of the NEB calculation; it should be a string of the structure file name
        image_number:the number of the images in the NEB calculation,not including the initial and final states;
        vasp_directory:the directory of the NEB calculation
        """
        from ase.neb import NEB
        from ase.optimize import LBFGS

        if not os.path.exists(vasp_directory):
            os.mkdir(vasp_directory)
            os.chdir(vasp_directory)
            # initial = read(intial_state)
            # final = read(final_state)
            images = [initial]
            images += [initial.copy() for i in range(image_number)]
            images += [final]
            print(images)
            neb = NEB(images,climb=True)
            if idpp:
                neb.interpolate('idpp',mic=True,apply_constraint=True)
            else:
                neb.interpolate(mic=True,apply_constraint=True)
            #neb.write('neb.traj')
            for image in images:
                tmp_dir = f'{vasp_directory}_{images.index(image)}'
                image.calc = Vasp(directory=tmp_dir,**kwargs)

            optimizer = LBFGS(neb, trajectory='A2B.traj')
            optimizer.run(fmax=0.08)
            os.chdir(os.path.dirname(os.getcwd()))



        



################################TEST###############################################
if __name__ == '__main__':
    """
    每一个实例化的work_flow对象都会在当前目录下生成一个以work_flow对象名字命名的文件夹，该文件夹下包含了scf、dos、band三个文件夹，分别用于存放scf、dos、band的计算结果。
    使用方法：work_flow对象名字.start()开始计算，work_flow对象名字.end()结束计算。中间插入计算步骤和相应数据处理方法，work_flow对象名字.dos()和work_flow对象名字.band()。
    """
    import os
    import sys
    from ase.io import read
    from ase.calculators.vasp import Vasp
    from work_flow_vasp import MyWorkFlow

    incar_para_MnCr = {'ncore':4,
                    'kpar':2,
                    'ispin':2,
                    'ldau_luj':{'Mn': {'L': 2, 'U': 3.9, 'J': 0},'O':{'L': -1, 'U': 0, 'J': 0},'Cr':{'L': 2, 'U': 3.7, 'J': 0}},
                    'lmaxmix' : 4,
                    'gamma':True,
                    'kpts':(2,2,2),
                    'xc':'PBE',
                    'setups':'recommended',
                    'lreal':'Auto',
                    'encut':550,
                    'ismear':0,
                    'sigma':0.1,
                    'ediff':1E-5,
                    'algo':'Veryfast',
                    'nelmin':10,
                    'nelm':400,
                    }
    incar_para_MnCrMg = {'ncore':4,
                    'kpar':2,
                    'ispin':2,
                    'ldau_luj':{'Mn': {'L': 2, 'U': 3.9, 'J': 0},'O':{'L': -1, 'U': 0, 'J': 0},'Cr':{'L': 2, 'U': 3.7, 'J': 0},'Mg':{'L': -1, 'U': 0, 'J': 0}},
                    'lmaxmix' : 4,
                    'gamma':True,
                    'kpts':(2,2,2),
                    'xc':'PBE',
                    'setups':'recommended',
                    'lreal':'Auto',
                    'encut':550,
                    'ismear':0,
                    'sigma':0.1,
                    'ediff':1E-5,
                    'algo':'Veryfast',
                    'nelmin':10,
                    'nelm':400,
                    }

    incar_para_list = [incar_para_MnCr,incar_para_MnCrMg]
    atoms, work_dir = MyWorkFlow.build_models_list_and_folders_name("*POSCAR*")
    for atom, workdir, incar_para in zip(atoms, work_dir,incar_para_list):
    #print(atom, workdir, incar_para)
        mywork = MyWorkFlow(atom, workdir,common_parameters=incar_para)
        #myworks.append(mywork)
        mywork.start()
        #mywork.start()
        #mywork.scf()
        #mywork.band(npoints=20,algo='Fast')
        mywork.check_job()
        
        #mywork.dos(algo='Fast')
        #mywork.get_plot(band=True)
        
        mywork.check_convergence()
        #print("scf_e",scf_e)
        mywork.end()