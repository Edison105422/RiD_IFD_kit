#!/usr/bin/env python3

import argparse
import json
import os
import sys
import numpy as np
from rid_ifd_kit.lib.make_ndx import make_ndx, make_protein_atom_index
from rid_ifd_kit.lib.utils import replace, print_list, print_repeat_list
from rid_ifd_kit.lib.make_def import make_general_angle_def, make_general_dist_def, make_angle_def, make_dist_def


def make_restraint(angle_names,
                   kappa,
                   at):
    ret = ""
    res_names = []
    for angle_print in angle_names:
        res_name = "res-" + angle_print
        res_names.append(res_name)
        ret += ((res_name + ":") + " " +
                "RESTRAINT " +
                ("ARG=" + angle_print).ljust(16) + " " +#"add".ljust(10,'0'): "add0000000"
                "KAPPA=" + str(kappa) + " " +
                "AT=" + str(at) + " " +
                "\n")
    return ret, res_names

def make_walls(dist_names,
                    kappa,#check papers for value of kappa
                    at_upper, at_lower,
                    exp,# assume the kappa and exp, eps are identical for up and low
                    eps,
                    offset):
    ret = ""
    wall_names = []
    for dist_print in dist_names:
        wall_name = "wall-" + dist_print
        wall_names.append("upper_" + wall_name)
        wall_names.append("lower_" + wall_name)
        ret += (("upper_" + wall_name + ":") + " " +
                "UPPER_WALLS " +
                ("ARG=" + dist_print).ljust(16) + " " +#"add".ljust(10,'0'): "add0000000"
                "AT=" + str(at_upper) + " " +
                "KAPPA=" + str(kappa) + " " +
                "EXP=" + str(exp) + " " +
                "EPS=" + str(eps) + " " +
                "OFFSET=" + str(offset) + " " +
                "\n" +
                ("lower_" + wall_name + ":") + " " +
                "LOWER_WALLS " +
                ("ARG=" + dist_print).ljust(16) + " " +#"add".ljust(10,'0'): "add0000000"
                "AT=" + str(at_lower) + " " +
                "KAPPA=" + str(kappa) + " " +
                "EXP=" + str(exp) + " " +
                "EPS=" + str(eps) + " " +
                "OFFSET=" + str(offset) + " " +
                "\n")
    return ret, wall_names

def make_deep_bias(angle_names,
                   trust_lvl_1=1.0,
                   trust_lvl_2=2.0,
                   model="graph.pb"):
    arg_list = print_list(angle_names)
    return ("dpfe: " +
            "DEEPFE " +
            "TRUST_LVL_1=" + str(trust_lvl_1) + " " +
            "TRUST_LVL_2=" + str(trust_lvl_2) + " " +
            "MODEL=" + model + " " +
            "ARG=" + str(arg_list) + " " +
            "\n")


def make_print(names,
               stride,
               file_name):
    arg_list = print_list(names)
    return ("PRINT " +
            "STRIDE=" + str(stride) + " " +
            "ARG=" + str(arg_list) + " " +
            "FILE=" + str(file_name) + " " +
            "\n")


def make_wholemolecules(atom_index):
    arg_list = print_list(atom_index)
    return ("WHOLEMOLECULES" + " " +
            "ENTITY0=" + arg_list +
            "\n")


def user_plumed(cv_file): #read from cv_file
    ret = "" # cat cv_file
    cv_names = []
    print_content = None #all content containing 'PRINT'
    with open(cv_file, 'r') as fp:
        for line in fp.readlines():
            if ("PRINT" in line) and ("#" not in line):
                print_content = line + "\n"
                break
            ret += line + "\n"
            if (":" in line) and ("#" not in line):
                cv_names.append(("{}".format(line.split(":")[0])).strip())
            elif ("LABEL" in line) and ("#" not in line):
                cv_names.append(("{}".format(line.split("LABEL=")[1])).strip())
    if ret == "" or cv_names == "":
        raise RuntimeError("Invalid customed plumed files.")
    if print_content is not None:
        assert len(print_content.split(",")) == len(cv_names), "There are {} CVs defined in the plumed file, while {} CVs are printed.".format(len(cv_names), len(print_content.split(",")) )
        print_content_list = print_content.split()
        print_content_list[-1] = "FILE=plm.out"
        print_content = " ".join(print_content_list)
    return ret, cv_names, print_content

def get_centerdist(CONF):
    dist_names = CONF
    return dist_names

def general_plumed(TASK,
                   CONF,
                   at_upper,
                   kappa=500.0,
                   temp=3000.0,
                   tau=10.0,
                   gamma=0.1,
                   pstride=5,
                   pfile="plm.out"):
    ret = ""
    if TASK == "res":
        ptr, ptr_names = make_restraint(cv_names, kappa, 0.0)
        ret += (ptr)
        ret += "\n"
    elif TASK == "upperwall":
        ptr, ptr_names = make_walls(get_centerdist(CONF), kappa, at_upper, 0.0, 2, 1, 0)
        ret += (ptr)
        ret += "\n"
    elif TASK == "dpbias":
        ret += (make_deep_bias(cv_names))
        ret += "\n"
    elif TASK == "bf":
        None
    else:
        raise RuntimeError("unknow task: " + TASK)
    return ret

def make_plumed(OUT,
                TASK,
                CONF,
                at_upper=5.0,
                kappa=500.0,
                temp=3000.0,
                tau=10.0,
                gamma=0.1,
                pstride=5,
                pfile="plm.out"):
    ret = general_plumed(TASK, CONF, at_upper, kappa=kappa, temp=temp,
                         tau=tau, gamma=gamma, pstride=pstride, pfile=pfile)
    if os.path.basename(OUT) == '':
        if TASK == "dpbias":
            OUT = os.path.abspath(OUT) + "/plumed.dat"
        elif TASK == "bf":
            OUT = os.path.abspath(OUT) + "/plumed.bf.dat"
        elif TASK == "res":
            OUT = os.path.abspath(OUT) + "/plumed.res.dat"
        elif TASK == "upperwall":
            OUT = os.path.abspath(OUT) + "/plumed.upperwall.dat"
    with open(OUT, 'w') as plu:
        plu.write(ret)

def make_plumed_origin(OUT,
                TASK,
                CONF,
                CV_FILE,
                at_upper,
                kappa=500.0,
                temp=3000.0,
                tau=10.0,
                gamma=0.1,
                pstride=5,
                pfile="plm.out"):
    ret = general_plumed(TASK, CONF, CV_FILE, at_upper, kappa=kappa, temp=temp,
                         tau=tau, gamma=gamma, pstride=pstride, pfile=pfile)
    if os.path.basename(OUT) == '':
        if TASK == "dpbias":
            OUT = os.path.abspath(OUT) + "/plumed.dat"
        elif TASK == "bf":
            OUT = os.path.abspath(OUT) + "/plumed.bf.dat"
        elif TASK == "res":
            OUT = os.path.abspath(OUT) + "/plumed.res.dat"
        elif TASK == "upperwall":
            OUT = os.path.abspath(OUT) + "/plumed.upperwall.dat"
    with open(OUT, 'w') as plu:
        plu.write(ret)

def conf_enhc_plumed(plm_conf, plu_type, #graph_list, enhc_trust_lvl_1=None, enhc_trust_lvl_2=None,
                     frame_freq=None, enhc_out_plm=None):
    """if plu_type == "enhc":
        replace(plm_conf, "MODEL=[^ ]* ", ("MODEL=%s " % graph_list))
        replace(plm_conf, "TRUST_LVL_1=[^ ]* ",
                ("TRUST_LVL_1=%f " % enhc_trust_lvl_1))
        replace(plm_conf, "TRUST_LVL_2=[^ ]* ",
                ("TRUST_LVL_2=%f " % enhc_trust_lvl_2))
        replace(plm_conf, "STRIDE=[^ ]* ", ("STRIDE=%d " % frame_freq))
        replace(plm_conf, "FILE=[^ ]* ", ("FILE=%s " % enhc_out_plm))
    """
    if plu_type == "bf":
        replace(plm_conf, "STRIDE=[^ ]* ", ("STRIDE=%d " % frame_freq))
        replace(plm_conf, "FILE=[^ ]* ", ("FILE=%s " % enhc_out_plm))


def make_res_templ_plumed(plm_path, conf_file,
                        at_upper, res_kappa, res_ang_stride, res_prt_file):
    ret = general_plumed("res", conf_file, at_upper, 
                         kappa=res_kappa,
                         pstride=res_ang_stride,
                         pfile=res_prt_file)
    if os.path.basename(plm_path) == "":
        plm_path = os.path.abspath(plm_path) + "/plumed.res.templ"
    with open(plm_path, "w") as fp:
        fp.write(ret)


def make_res_templ_plumed_origin(plm_path, conf_file, cv_file, 
                        at_upper, res_kappa, res_ang_stride, res_prt_file):
    ret = general_plumed("res", conf_file, cv_file, at_upper, 
                         kappa=res_kappa,
                         pstride=res_ang_stride,
                         pfile=res_prt_file)
    if os.path.basename(plm_path) == "":
        plm_path = os.path.abspath(plm_path) + "/plumed.res.templ"
    with open(plm_path, "w") as fp:
        fp.write(ret)


def conf_res_plumed(plm_path, frame_freq):
    replace(plm_path, "STRIDE=[^ ]* ", "STRIDE=%d " % frame_freq)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK", type=str,
                        help="the type of task, either res, afed or dpbias")
    parser.add_argument("CONF", type=str,
                        help="the conf file")
    #parser.add_argument("CVFILE", type=str,
    #                    help="the json file defining the dih angles")
    parser.add_argument("-a", "--at_upper", default=5.0, type=float,
                        help="the upperwall limit")
    parser.add_argument("-k", "--kappa", default=500, type=float,
                        help="the spring constant")
    parser.add_argument("-T", "--temp", default=3000.0, type=float,
                        help="the temperature of afed")
    parser.add_argument("-t", "--tau", default=10.0, type=float,
                        help="the relaxation timescale of afed")
    parser.add_argument("-g", "--gamma", default=0.1, type=float,
                        help="the frection const of afed")
    parser.add_argument("-s", "--stride", default=5, type=int,
                        help="the printing stride")
    parser.add_argument("-f", "--print-file", default="plm.out", type=str,
                        help="the printing file")
    args = parser.parse_args()

    ret = general_plumed(args.TASK, args.CONF, args.at_upper, args.kappa,
                         args.temp, args.tau, args.gamma, args.stride, args.print_file)

    print(ret)


if __name__ == '__main__':
    # _main()
    CONF = "/home/dongdong/wyz/rid-kit/tests/benchmark_mol/conf.gro"
    CV_FILE = "/home/dongdong/wyz/rid-kit/tests/benchmark_json/cv.json"
    TASK = "dpbias"
    general_plumed(TASK, CONF, at_upper=5.0, kappa=500.0, temp=3000.0,
                   tau=10.0, gamma=0.1, pstride=5, pfile="plm.out")

def general_plumed_origin(TASK,
                   CONF,
                   CV_FILE,
                   at_upper,
                   kappa=500.0,
                   temp=3000.0,
                   tau=10.0,
                   gamma=0.1,
                   pstride=5,
                   pfile="plm.out"):
    if CV_FILE.split(".")[-1] == "json":
        with open(CV_FILE, 'r') as fp:
            jdata = json.load(fp)
        residues, residue_atoms = make_ndx(CONF)
        protein_atom_idxes = make_protein_atom_index(CONF)
        dih_angles = jdata["dih_angles"]
        fmt_alpha = jdata["alpha_idx_fmt"]
        fmt_angle = jdata["angle_idx_fmt"]
        hp_residues = []
        dist_atom = []
        dist_excl = 10000
        if "hp_residues" in jdata:
            hp_residues = jdata["hp_residues"]
        if "dist_atom" in jdata:
            dist_atom = jdata["dist_atom"]
        if "dist_excl" in jdata:
            dist_excl = jdata["dist_excl"]

        angle_names, angle_atom_idxes = make_general_angle_def(
            residue_atoms, dih_angles, fmt_alpha, fmt_angle)

        dist_names, dist_atom_idxes = make_general_dist_def(
            residues, residue_atoms, hp_residues, dist_atom, fmt_alpha, dist_excl)

        if "selected_index" in jdata:
            selected_index = jdata["selected_index"]
            selected_angle_index = []
            for ssi in selected_index:
                if ssi == 0:
                    selected_angle_index.append(0)
                elif ssi != 0:
                    selected_angle_index.append(ssi*2-1)
                    selected_angle_index.append(ssi*2)

            newselected_angle_index = [
                i for i in selected_angle_index if i < len(angle_names)]
            new_angle_names = [angle_names[i] for i in newselected_angle_index]
            new_angle_atom_idxes = [angle_atom_idxes[i]
                                    for i in newselected_angle_index]
        else:
            new_angle_names = angle_names
            new_angle_atom_idxes = angle_atom_idxes

        ret = ""

        if len(dist_names) > 0:
            ret += make_wholemolecules(protein_atom_idxes)
            ret += "\n"
        ret += (make_angle_def(new_angle_names, new_angle_atom_idxes))
        ret += (make_dist_def(dist_names, dist_atom_idxes))
        ret += "\n"

        cv_names = new_angle_names + dist_names
        print_content = None

    else:
        ret, cv_names, print_content = user_plumed(CV_FILE)

    if TASK == "res":
        ptr, ptr_names = make_restraint(cv_names, kappa, 0.0)
        ret += (ptr)
        ret += "\n"
    elif TASK == "upperwall":
        ptr, ptr_names = make_walls(cv_names, kappa, at_upper, 0.0, 2, 1, 0)
        ret += (ptr)
        ret += "\n"
    elif TASK == "dpbias":
        ret += (make_deep_bias(cv_names))
        ret += "\n"
    elif TASK == "bf":
        None
    else:
        raise RuntimeError("unknow task: " + TASK)
    
    if print_content is None:
        ret += (make_print(cv_names, pstride, pfile))
        ret += "\n"
    else:
        ret += print_content
    return ret
