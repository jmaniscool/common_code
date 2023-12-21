# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:15:18 2023

Decorator code to make good figures. A direct upgrade to mpltex which relies moreso on matplotlib v3, ensuring it will be more future-proof.

Code framework and some options are inspired by mpltex https://github.com/liuyxpp/mpltex.
Much of the code has been overhauled to be more future-proof, and to make more customizeable if one desires in the future.

v1:
    --uses xelatex to generate figures instead of matplotlib's pdftex, allowing for more customizability when creating figures.
    --uses open-source Calibri instead of mpltex's closed-source Helvetica, a nice looking open source font compatible with Adobe Illustrator
    --uses OpenType Latin Modern font for equations, which future proofs equations as compared with mpltex.
    --makes square plots, which may be more visually appealing.
    
v2:
    --Automatically rasterize plots with more than 100,000 datapoints in them (?)


#SETUP INSTRUCTIONS:
    1) Install a working version of LaTeX (Tested working with Tex Live https://www.tug.org/texlive/ and possibly with MiKTeX)
    2) Install the optional package unicode-math in LaTeX. I recommend using the TeX editor TeXStudio, since it automatically installs all missing packages. If you must install manually, follow https://www.youtube.com/watch?v=6N2rjNw0YOs&ab_channel=Electricallectures.
    3) Install all of the Latin Modern OpenType fonts as system fonts
        3a) Latin Modern font family: https://www.gust.org.pl/projects/e-foundry/latin-modern
        3b) Latin Modern math: https://ctan.org/tex-archive/fonts/lm-math?lang=en
    4) Remove the matplotlib font cache
        --obtain the directory of the font cache by importing matplotlib then running print(matplotlib.get_cachedir())
        --remove the font cache. For me, it's called fontlist-v330.json
    5) Once removed, matplotlib will generate the font cache automatically when a plot is made. The fonts are populated from your OS's font install directory.

@author: sickl
"""

from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from collections.abc import Iterable #to check if object is iterable
import numpy as np

#palettable is a pure python library that holds a few nice color palettes
from palettable.tableau import Tableau_10 #define default colors as tableau (blue to orange to green)
from cycler import cycler #use cycler library to do cycle


#####COLORS#####
#code adopted from mpltex
# Set some commonly used colors
almost_black = '#262626'
light_grey = np.array([float(248) / float(255)] * 3)

tableau_10 = Tableau_10.mpl_colors
# Add a decent black color
tableau_10.append(almost_black)
# Swap orange and red
tableau_10[1], tableau_10[3] = tableau_10[3], tableau_10[1]
# swap orange and purple
# now table_au has similar sequence to brewer_set1
tableau_10[3], tableau_10[4] = tableau_10[4], tableau_10[3]

mpltex_cycler = cycler('color', tableau_10)


#using optimal colors for accessibility, see http://tsitsul.in/blog/coloropt/ for details
#I don't like how this color scheme looks, though, so I prefer tableau
class optimal_colors:
    blue = '#4053d3'
    yellow = '#ddb310'
    red = '#b51d14'
    lblue = '#00beff'
    pink = '#fb49b0'
    green = '#00b25d'
    gray = '#cacaca'
    grey = '#cacaca' #for alternate spelling of gray.

#12 optimal colors, from same source, if more colors are required.
class optimal_colors12:
    color_codes = [(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)]
    yellow = tuple([i/255 for i in color_codes[0]])
    lipstick = tuple([i/255 for i in color_codes[1]])
    azure = tuple([i/255 for i in color_codes[2]])
    green = tuple([i/255 for i in color_codes[3]])
    caribbean = tuple([i/255 for i in color_codes[4]])
    lavender = tuple([i/255 for i in color_codes[5]])
    brown = tuple([i/255 for i in color_codes[6]])
    coral = tuple([i/255 for i in color_codes[7]])
    indigo = tuple([i/255 for i in color_codes[8]])
    turquoise = tuple([i/255 for i in color_codes[9]])
    olive = tuple([i/255 for i in color_codes[10]])
    jade = tuple([i/255 for i in color_codes[11]])
    gray = tuple([i/255 for i in color_codes[12]])
    grey = tuple([i/255 for i in color_codes[12]])



opt_cycle = [optimal_colors.blue, optimal_colors.red, optimal_colors.green, optimal_colors.yellow, optimal_colors.lblue, optimal_colors.pink]

opt_cycler = cycler('color',opt_cycle)

default_cycler = mpltex_cycler



###############

#####LAYOUT#####
GOLDEN_RATIO = (1 + np.sqrt(5))/2

width_single_column = 3.375 #set by aps figure guidelines
width_double_column = 6.75

# Default ratio for a single plot figure
#height_width_ratio = GOLDEN_RATIO * 1.1  # The height/width ratio preferred by mpltex.
height_width_ratio = 1 #the height/width ratio preferred by style guide https://www.mrl.ucsb.edu/~seshadri/PreparingFigures-June2019.pdf

#some more guidance on how to remove clutter found at http://www.perceptualedge.com/articles/visual_business_intelligence/the_chartjunk_debate.pdf
#some more guidance on making figures found at https://mcmanuslab.ucsf.edu/sites/mcmanuslab.ucsf.edu/files/event/file-attachments/design-tips-scientistsguide.pdf

#font sizes
smallfont = 7
medfont = 9
largefont = 16

#text properties for subpanel
subpanel_text_prop = FontProperties(family='Calibri', weight='bold', size = medfont)

_width = width_single_column
_height = width_single_column * height_width_ratio

#helper function which converts a number in decimal to a simple LaTeX readable string using the
#functions defined in the decorator below. That is, \ss is the \textsuperscript function.
def convert_to_times(num,sigfigs = 0):
    formatter = "{:.%de}" % (sigfigs) #custom sig figs
    val = formatter.format(num)
    eidx = val.index("e")
    
    mantissa = float(val[:eidx])
    
    #OLD OPTION. This looks bad when everything else is in scientific notation!
    #if val[0] is between 1 and 10, display as is
    #if (num > 1)*(num < 10):
    #    return val[:eidx]
    
    #if mantissa is 1 exactly, then make the number in the format 10^{b}
    if mantissa == 1:
        val2 = val.replace("e",r"10\ss{").replace("+0","").replace("-0","-") + "}"
        #val2 = val2.replace("10\ss{","10\ss{ ")
        val2 = val2[eidx:]
        return val2
    #if val[0] is not 1 and is not between 1 and 10, then make the number in the format a \times 10^{b}
    val2 = val.replace("e",r" \texttimes 10\ss{").replace("+0","").replace("-0","-") + "}"    
    #val2 = val2.replace("10\ss{","10\ss{ ")
    return val2


#to use units, just have to specify units and spell them out with macros.
#Example kg m/s^2 = \unit{\kilo\gram\meter\per\square\second}
#you could also just write out the unit and the \unit command will parse.
#Example kg m/s^2 = \unit{kg.m/s^{2}}
#might need to set pgf.texsystem to xelatex, but this is the default.
def figure_decorator(fontsize = medfont):
    subpanel_text_prop = FontProperties(family='Calibri', weight='bold', size = fontsize)
    def figure_decorator_wrap(func):
        def wrapper(*args, **kwargs):
            
            #axes
            rcParams['axes.prop_cycle'] = default_cycler
            rcParams['axes.linewidth'] = 1 #line width equals 1 is consistent.
            rcParams['axes.xmargin'] = 0.05 #fraction of figure left as 'buffer' around data on x axis. Default is 0.1.
            rcParams['axes.ymargin'] = 0.05 #fraction of figure left as 'buffer' around data on y axis. Default is 0.1.
            
            titlesize = fontsize
            axessize = fontsize
            ticksize = fontsize
            legendsize = fontsize
            
            #save options
            rcParams['figure.figsize'] = (_width,_height) #width and height set for single
            rcParams['savefig.format'] = 'pdf'
            rcParams['savefig.dpi'] = 1800
            
            
            #legend
            rcParams['legend.fontsize'] = legendsize #font size for legend
            rcParams['legend.frameon'] = False #turn off the frame for the legend
            rcParams['legend.numpoints'] = 1 #number of points used when making a legend for a line plot. Default.
            rcParams['legend.handlelength'] = 1 #the length of the handle (i.e. the example symbol) compared to the default value of 1.
            rcParams['legend.scatterpoints'] = 1 #the number of points that are used to create the handle for a scatterpoint legend. Default is 1.
            rcParams['legend.labelspacing'] = 0.5 #The vertical space between the legend entries, in font-size units. default is 0.5
            rcParams['legend.markerscale'] = 1 #The relative size of legend markers compared to the originally drawn ones. Default is 1.
            rcParams['legend.handletextpad'] = 0.35  # pad between handle and text. Default is 0.8, mpltex uses 0.5.
            rcParams['legend.borderaxespad'] = 0.5  # pad between legend and axes. Default is 0.5. mpltex uses 0.5
            rcParams['legend.borderpad'] = 0.5  # pad between legend and legend content. Default is 0.5, mpltex uses 0.5.
            rcParams['legend.columnspacing'] = 0.75  # pad between each legend column. Default is 2.0, mpltex uses 1.
            
            #latex
            rcParams['backend'] = 'pgf' #allows xelatex to parse the preamble. Required for unicode-math
            rcParams['pgf.texsystem'] = 'xelatex' #sets the texsystem to be xelatex
            rcParams['pgf.preamble'] = r'''
            \usepackage{unicode-math}
            \setmainfont{Calibri}
            \setmathfont{Latin Modern Math}
            \setsansfont{Calibri}
            \usepackage{siunitx}
            \usepackage{textcomp}
            \sisetup{detect-all}
            \pagenumbering{gobble}
            \renewcommand{\^}[1]{\textsuperscript{#1}}
            \renewcommand{\ss}[1]{\textsuperscript{#1}}
            \renewcommand{\_}[1]{\textsubscript{#1}}
            '''
            rcParams['font.sans-serif'] = 'Calibri' #set the sans-sarif font default to Calibri in matplotlib
            
            #title
            rcParams['axes.titlesize'] = titlesize
            
            #axis label
            rcParams['axes.labelsize'] = axessize #set the default label size of the axes
            
            #ticks
            rcParams['xtick.labelsize'] = ticksize
            rcParams['xtick.direction'] = 'in' #point ticks inwards
            rcParams['xtick.top'] = True
            rcParams['xtick.major.pad'] = 8*(fontsize/12) #change the padding between the x tick labels and the axis. Default is 3.5
            
            rcParams['ytick.labelsize'] = ticksize
            rcParams['ytick.direction'] = 'in'
            rcParams['ytick.right'] = True
            rcParams['ytick.major.pad'] = 8*(fontsize/12) #change the padding between the y tick labels and the axis. Default is 3.5
            
            fig,axs = func(*args, **kwargs)
            
            
            #allow either list of axs input or just a single axis object
            already_iterable = True
            if isinstance(axs,Iterable) == False:
                axs = np.array([axs])
                already_iterable = False
            
            orig_shape = axs.shape #store the original shape
            flataxs = axs.flatten() #flatten the multidimensional array into a single list
            for i in range(len(flataxs)):
                ax = flataxs[i]
                ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)  # Set x-axis label font size
                ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)  # Set y-axis label font size
                
                
                #if the xscale is log, then manually change each x axis label to be in the non-math font.
                if ax.get_xscale() == 'log':
                    
                    x0,x1 = ax.get_xlim() #get the current x limit
                    
                    #only change the x tick labels that are currently visible
                    xtick_labels = [t for t in ax.get_xticklabels() if t.get_position()[0]>= x0 and t.get_position()[0] <= x1]
                    
                    vals = []
                    newlabels = []
                    for xtick in xtick_labels:
                        val = xtick.get_position()[0] #get x position of x tick
                        newlabel = convert_to_times(val)
                        vals.append(val)
                        newlabels.append(newlabel)
                    ax.set_xticks(vals,newlabels)
                        
                if ax.get_yscale() == 'log':
                    
                    y0,y1 = ax.get_ylim() #get the current y limit
                    
                    #only change the y tick labels that are currently visible
                    ytick_labels = [t for t in ax.get_yticklabels() if t.get_position()[1] >= y0 and t.get_position()[1] <= y1]
                    vals = []
                    newlabels = []
                    for ytick in ytick_labels:
                        val = ytick.get_position()[1] #get y position of y tick
                        newlabel = convert_to_times(val)
                        vals.append(val)
                        newlabels.append(newlabel)
                    ax.set_yticks(vals,newlabels)
            
            #reshape the axes into their original shape
            axs = flataxs.reshape(orig_shape)
            
            #if the object was not iterable to begin with, turn it back to a non-iterable object.
            if already_iterable == False:
                axs = axs[0]
            
            return fig,axs
        return wrapper
    return figure_decorator_wrap

#testing the stuff out.

r"""
@decorator(fontsize = 9) #plot function
def fun():
    fig, ax = plt.subplots() #always use constrained option
    ax.plot([1, 2, 3], [4, 5, 6], label = 'my first label')
    ax.plot([2,2,2],[5,4,5], '--', label = 'my second label')
    ax.set_xlabel(r'Hello world (\unit{\micro\ohm})')
    ax.set_ylabel(r'Hello y-axis')
    plt.tight_layout()    
    
    #set the subpanel text
    ax.annotate('(a)', xy=(0.05, 0.95), xycoords='axes fraction', fontproperties = subpanel_text_prop,
            horizontalalignment='left', verticalalignment='top')
    ax.legend() #follow advice on https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot to get the legend outside of a plot, which may enhance legibility.
    return fig,ax

fig,ax = fun()

fig.savefig('figure.pdf', backend='pgf')

plt.close()
"""