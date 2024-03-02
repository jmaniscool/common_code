#test
from .get_slips import get_slips_wrap as gs
from .fit import fit as fit
from .shapes import shapes
from .weinerfilter import weinerfilter_wrap as weinerfilter
from .logbinning import logbinning
#from .scaling_gui_v3 import scaling_gui #defunct for now
from .get_ccdf_arr import get_ccdf_arr as ccdf
from .ccdf_errorbar import ccdf_errorbar
from .decorator import figure_decorator
from .generate_noise import generate_noise

from .autofilter import autofilter

from .culling import culling

#constants
from .decorator import smallfont
from .decorator import medfont
from .decorator import largefont
from .decorator import subpanel_text_prop
from .decorator import convert_to_times #just in case you need to tune for a figure
from .decorator import optimal_colors
from .decorator import optimal_colors12

from .likelihoods import find_pl_exact as find_pl_exact
from .likelihoods import find_pl as find_pl
from .likelihoods import find_tpl as find_tpl
from .likelihoods import find_exp
from .likelihoods import find_lognormal
from .likelihoods import llr_wrap as llr
from .likelihoods import bootstrap
from .likelihoods import bca
from .likelihoods import bootstrap2 as bootstrap_test
from .likelihoods import ad
from .likelihoods import pl_gen
from .likelihoods import tpl_gen
from .likelihoods import find_pl_montecarlo as find_pl_montecarlo