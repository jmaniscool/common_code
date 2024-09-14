
from .get_slips import get_slips_wrap as gs
from .get_slips import get_slips_vel as gsv #get_slips to use if you have a velocity signal and you want to guarantee no negative sizes.
from .fit import fit as fit
from .shapes import shapes
from .weinerfilter import weinerfilter_wrap as weinerfilter
from .logbinning import logbinning
#from .scaling_gui_v3 import scaling_gui #defunct for now
from .get_ccdf_arr import ccdf as ccdf #Ethan's updated ccdf function which works better for larger data.
from .get_ccdf_arr import ccdf_unique as ccdf_unique #New function for if unique values in the CCDF are found
from .ccdf_errorbar import ccdf_errorbar
from .decorator import figure_decorator
from .generate_noise import generate_noise

from .autofilter import autofilter
from .autofilter import butter_highpass_filter as highpass

from .culling import culling
from .linemaker import linemaker #ethan's linemaker function, added 6-20-24

#constants
from .decorator import smallfont
from .decorator import medfont
from .decorator import largefont
from .decorator import subpanel_text_prop
from .decorator import convert_to_times #just in case you need to tune for a figure
from .decorator import optimal_colors
from .decorator import optimal_colors12

#likelihoods (contains likelihoods, generator functions, and relative testing of distributions.)
from .likelihoods import find_pl
from .likelihoods import find_pl_discrete
from .likelihoods import find_tpl
from .likelihoods import find_exp
from .likelihoods import find_lognormal_truncated as find_lognormal
from .likelihoods import llr_wrap as llr
from .likelihoods import ad
from .likelihoods import pl_gen
from .likelihoods import pl_gen_discrete
from .likelihoods import tpl_gen
from .likelihoods import lognormal_gen
from .likelihoods import generate_test_data_with_xmax as generate_test_data

#bootstrap (contains just bootstrap functions and all helpers)
from .bootstrap import bootstrap as bootstrap
from .bootstrap import bca
from .bootstrap import bootstrap_bca

#montecarlo (contains find_pl_montecarlo)
from .montecarlo import find_pl_montecarlo as find_pl_montecarlo
from .montecarlo import find_p_wrap as find_p
from .montecarlo import find_p_discrete_wrap as find_p_discrete

#functions that aren't used any more are contained in defunct.py, but are kept for posterity.
