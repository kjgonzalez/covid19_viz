'''
author: kjgonzalez
objective: create a graph per-country of population-percentage vs days since first infection

quick note about plotting:
7 diff colors: b / g / r / c / m / y / k
4 diff linestyles: - / -- / -. / :
9 diff markers: <none> / , / o / v / ^ / s / * / + / x

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, argparse
from scipy.interpolate import make_interp_spline, BSpline

def listContents(arr,ReturnAsNPArr=False):
    ''' Take in a string list and return a dict or numpy array of
    the unique values and the number of times they appear.
    '''
    z=dict()
    for _irow in arr:
        if(_irow in z.keys()):
            z[_irow]+=1 # already seen, increment counter
        else:
            z[_irow]=1 # irow never seen, create key and set to 1
    if(ReturnAsNPArr==False):
        return z
    else:
        _items=np.array([list(z.keys())],dtype='object').transpose()
        nums=np.array([list(z.values())],dtype='object').transpose()
    return np.column_stack((_items,nums))

def dayssince(_dates,_fmt='%m/%d/%Y'):
    seconds = np.array([time.mktime(time.strptime(i+'20',_fmt)) for i in _dates])
    return np.round(seconds/(60*60*24))


def parse_countries_string(countries_string,all_locations):
    ''' separate string into list of country names '''
    csplit = lambda s:s.replace('_',' ').split('-')
    cs = csplit(countries_string)
    if('all' in cs):
        # quick solution, exception to regular parsing
        return all_locations

    if('eu' in cs):
        cs.remove('eu')
        cs+=csplit(C_EU)
        cs=list(set(cs))
    if('samerica' in cs):
        cs.remove('samerica')
        cs+=csplit(C_SA)
    if('namerica' in cs):
        cs.remove('namerica')
        cs+=csplit(C_NA)
    return cs

# some constants
base = 'COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_covid19_{}_global.csv'
f_con = base.format('confirmed')
f_dea = base.format('deaths')
f_rec = base.format('recovered')
f_pop = 'populations.csv'

C_EU = 'Austria-Belgium-Bulgaria-Croatia-Cyprus-Czechia-' + \
     'Denmark-Estonia-Finland-France-Germany-Greece-Hungary-Ireland-Italy-' + \
     'Latvia-Lithuania-Luxembourg-Malta-Netherlands-Poland-Portugal-Romania-' + \
     'Slovakia-Slovenia-Spain-Sweden'
C_SA = 'Argentina-Bolivia-Brazil-Chile-Colombia-Ecuador-French_Guiana-'+\
           'Guyana-Paraguay-Peru-Suriname-Uruguay-Venezuela'
C_NA = 'US-Mexico-Canada-Guatemala-Cuba-Haiti-Dominican_Republic-Honduras-Nicaragua-'+\
           'El_Salvador-Costa_Rica-Panama-Puerto_Rico-Jamaica-Trinidad_and_Tobago-Bahamas-'+\
           'Guadeloupe-Martinique-Belize-Barbados-Saint_Lucia-Antigua_and_Barbuda-Grenada'
C_AA = 'China-India-Indonesia-Pakistan-Bangladesh-Russia-Japan-Phillippines-Vietnam-'+\
       'Thailand-Myanmar-Korea,_South-Afghanistan-Uzbekistan-Malaysia-'+\
       'Nepal-Korea,_North-Taiwan*-Sir_Lanka-Kazakhstan-Cambodia-Azerbaijan'
C_ME = 'Egypt-Iran-Turkey-Iraq-Saudi_Arabia-Syria-Yemen-United_Arab_Emirates-Israel-Jordan-'+\
       'Palestine-Lebanon-Kuwait-Oman-Qatar-Bahrain'

# check that there's no overlap between regions
_all = '-'.join([C_EU,C_SA,C_NA,C_AA,C_ME])
_all = parse_countries_string(_all,_all)
_res = listContents(_all,True)
_whereGreater = np.where(_res[:,1]>1)
assert _res[:,1].max()==1,"repeated locations:\n{}".format(_res[_whereGreater])

# generate all possible combinations, with priority to color & marker
formats = []
for ils in '- -- -. :'.split(' '):
    for imk in ' , o * v x ^ s +'.split(' '):
        for iclr in 'bgrcmyk':
            formats.append( iclr+ils+imk )

class Country:
    def __init__(self,name,days,values,pop,thresh=0,smooth=False,formatPlot=''):
        self.name=name
        self.days=days
        self.vals=values
        self.pop=pop
        self._thresh=thresh
        self.fmtplot=formatPlot
        try:
            self.ind_first=np.where(values>thresh)[0][0]
        except IndexError:
            # this means values have not passed threshold. simply plot last two values
            self.ind_first=len(values)-2

        if(smooth):
            self.days = np.linspace(days.min(), days.max(),len(days)*5)
            spl=make_interp_spline(days, values, k=3)  # type: BSpline
            self.vals = spl(self.days)

    @property
    def _all(self):
        return (self.name,self.days,self.vals)

    @property
    def percentpop(self):
        return self.vals/self.pop*100

    @property
    def daysAdj(self):
        # import ipdb; ipdb.set_trace()
        return self.days[self.ind_first:]-self.days[self.ind_first]

    @property
    def valsAdj(self):
        return self.vals[self.ind_first:]
    @property
    def percentpopAdj(self):
         return self.percentpop[self.ind_first:]
    @property
    def per100k(self):
        return self.percentpop/100*100e3
    @property
    def per100kAdj(self):
        return self.percentpopAdj/100*100e3

    @property
    def newvals(self):
        return self.days,self.vals

if(__name__=='__main__'):
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--src',default='confirmed',help='Confirmed, Deaths, Recovered')
    p.add_argument('--thresh',default=50,type=int,help='min number of cases for aligning chronologically')
    p.add_argument('--locs',default='China-US-Italy-Germany',
                   help='desired countries. Examples: United_Kingdom-China')
    p.add_argument('--smooth',default=False,action='store_true',help='create smooth curves')
    p.add_argument('--topAbs',default=None,type=int,help='top N countries by absolute number')
    p.add_argument('--botAbs',default=None,type=int,help='bottom N countries by absolute number')
    p.add_argument('--minus',default='Fiji-Diamond_Princess',help='excluded countries.')
    args=p.parse_args()
    if(args.topAbs is not None and args.botAbs is not None):
        raise Exception("cannot currently plot both top and bottom countries")

    # load raw data
    pdcon = pd.read_csv(base.format(args.src))
    dates=np.array(pdcon.columns[4:])
    fmt='%m/%d/%Y'
    fmt2='%Y%b%d'
    date0=time.strftime(fmt2,time.strptime(dates[0]+'20',fmt)) # start of worldwide outbreak
    dcon=np.array(pdcon)
    days = dayssince(dates)
    days = days-days[0] # days since start of worldwide outbreak
    locs_affected = listContents(dcon[:,1],True)[:,0] # locs_affected of affected countries
    locs_affected = locs_affected[np.argsort(locs_affected)] # sort locs_affected alphabetically
    pops = np.array(pd.read_csv(f_pop,header=None))
    pops = { i[0]:int(i[1]) for i in pops } # overwrite as dictionary

    # load locs and minus, remove minus, remove locations not affected
    LOCS = parse_countries_string(args.locs,locs_affected)
    _MINUS = parse_countries_string(args.minus,locs_affected)
    LOCS = list((set(LOCS) - set(_MINUS)).intersection(set(locs_affected)))
    LOCS.sort() # help keep colors consistent across runs
    # can only filter by top / bottom N locations AFTER data loading is complete

    # check that all values in LOCS are in population data
    for iloc in LOCS:
        if(iloc not in pops.keys()):
            ipop=input('Country not in pops.csv. Please give population for '+
                       iloc+' (will be saved): ')
            f=open(f_pop,'a')
            f.write('"{}",{}\n'.format(iloc,ipop))
            f.close()

    # reload population data
    pdpops = np.array(pd.read_csv(f_pop,header=None))
    pops = { i[0]:int(i[1]) for i in pdpops }

    # put yvalues into single array
    numbers=np.zeros((len(locs_affected),len(dcon[0,4:]))).astype(int)
    for irow in dcon:
        ind = np.where(irow[1]==locs_affected)
        numbers[ind]+=irow[4:].astype(int)
    numbers=numbers.astype(int)

    # put all data into dict of objects
    data = dict()
    plotlist=np.array(LOCS)
    for i,iname in enumerate(plotlist):
        ind = np.where(iname==locs_affected)[0][0]
        data[iname]=Country(iname,days,numbers[ind],pops[iname],
                            thresh=args.thresh,smooth=args.smooth,formatPlot=formats[i])

    # order the list in ascending abs
    vals = np.array([data[iname].vals.max() for iname in plotlist])
    mask=np.argsort(vals)
    plotlist=plotlist[mask[::-1]]
    vals=vals[mask[::-1]]
    # filter out for top / bottom N countries
    if(args.topAbs!=None):
        plotlist=plotlist[:args.topAbs]
    elif(args.botAbs!=None): plotlist = plotlist[::-1][:args.botAbs]






    # plot everything
    print('number of affected countries / sovereignties:',len(locs_affected))
    print('plotting {} countries...'.format(len(plotlist)))



    f,p = plt.subplots(figsize=(10,7))
    lines=[]
    for j,iloc in enumerate(plotlist):
        lines.append( p.plot(data[iloc].daysAdj, data[iloc].per100kAdj, data[iloc].fmtplot,
                             label='{} ({:.3}m)'.format(iloc, pops[iloc]/1e6),linewidth=3)[0] )
    p.set_ylabel('Cases/100,000 [Count]')
    p.set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
    f.suptitle('''{} Cases per 100,000 people
    Adjusted for start of outbreak (thresh:{})'''.format(args.src,args.thresh))
    p.grid()
    p.legend()
    annot1=p.annotate("", xy=(0,0), xytext=( 40,-20 ), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)

    def update_annot(name, xmouse, ymouse):
        annot1.xy = (xmouse, ymouse)
        annot1.set_text(name)
        annot1.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot1.get_visible()
        if event.inaxes == p:
            _ind = -1
            for i in range(len(lines)):
                if (lines[i].contains(event)[0]):
                    _ind = i
            if (_ind > -1):
                update_annot(lines[_ind]._label, event.xdata, event.ydata)
                annot1.set_visible(True)
                f.canvas.draw_idle()
            elif (vis):
                annot1.set_visible(False)
                f.canvas.draw_idle()
    f.canvas.mpl_connect("motion_notify_event", hover)

    # lines2=[]
    # f2, p2 = plt.subplots(figsize=(10,7))
    # for j, iloc in enumerate(plotlist):
    #     lines2.append( p2.plot(data[iloc].daysAdj, data[iloc].valsAdj,
    #             formats[j],
    #             label='{} ({})'.format(iloc, data[iloc].valsAdj.max()/1e3),
    #             linewidth=3)[0]
    #
    #            )
    # p2.set_ylabel('Total Cases [Count]')
    # p2.set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
    # f2.suptitle('''Total {} Cases
    # Adjusted for start of outbreak (thresh:{})'''.format(args.src, args.thresh))
    # p2.grid()
    # p2.set_yscale('log')
    # p2.legend()
    # annot2=p2.annotate("", xy=(0,0), xytext=( 40,-20 ), textcoords="offset points",
    #                 bbox=dict(boxstyle="round", fc="w"),
    #                 arrowprops=dict(arrowstyle="->"))
    # annot2.set_visible(False)
    #
    # def update_annot2(name, xmouse, ymouse):
    #     annot2.xy = (xmouse, ymouse)
    #     annot2.set_text(name)
    #     annot2.get_bbox_patch().set_alpha(0.4)
    #
    # def hover2(event):
    #     vis = annot2.get_visible()
    #     if event.inaxes == p2:
    #         _ind = -1
    #         for i in range(len(lines2)):
    #             if (lines2[i].contains(event)[0]):
    #                 _ind = i
    #         if (_ind > -1):
    #             update_annot2(lines2[_ind]._label, event.xdata, event.ydata)
    #             annot2.set_visible(True)
    #             f2.canvas.draw_idle()
    #         elif (vis):
    #             annot2.set_visible(False)
    #             f2.canvas.draw_idle()
    # f2.canvas.mpl_connect("motion_notify_event", hover2)
    #
    # # for now, only plot the first (highest) entry
    # f3,p3=plt.subplots()
    # for iname in plotlist:
    #     ydiff = data[iname].valsAdj[1:]-data[iname].valsAdj[:-1]
    #     p3.bar(data[iname].daysAdj[:-1],ydiff,label=iname)
    # p3.legend()
    # p3.grid()
    # # show plots
    plt.show()
    print('done')
