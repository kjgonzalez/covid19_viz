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
new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
formats = []
for ils in '- -- -. :'.split(' '):
    for imk in ' o * v x ^ s , +'.split(' '):
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
    def _percentpop(self):
        return self.vals/self.pop*100

    @property
    def _daysAdj(self):
        # import ipdb; ipdb.set_trace()
        return self.days[self.ind_first:]-self.days[self.ind_first]

    @property
    def _valsAdj(self):
        return self.vals[self.ind_first:]
    @property
    def _percentpopAdj(self):
         return self._percentpop[self.ind_first:]
    @property
    def _per100k(self):
        return self._percentpop/100*100e3
    @property
    def _per100kAdj(self):
        return self._percentpopAdj/100*100e3

    @property
    def D_raw(self):
        return self.days,self.vals,self.fmtplot

    @property
    def D_dateAdjusted(self):
        return self._daysAdj,self._valsAdj,self.fmtplot

    @property
    def D_dateAdjPer100k(self):
        return self._daysAdj,self._per100kAdj,self.fmtplot

    @property
    def D_dateAdjNewDaily(self):
        ydiff = entity[iloc]._valsAdj[1:] - entity[iloc]._valsAdj[:-1]
        return entity[iloc]._daysAdj[:-1], ydiff, self.fmtplot

class PlotObject:
    ''' want to make simple object that sets up graph '''
    def __init__(self,title='', xlabel='', ylabel='', scale='linear'):
        self.f,self.p=plt.subplots(figsize=(10,7))
        self.curves = []
        self.p.set_ylabel(ylabel)
        self.p.set_xlabel(xlabel)
        self.f.suptitle(title)
        self.p.grid()
        self.p.set_yscale(scale)
        self.f.canvas.mpl_connect("motion_notify_event", self._hover)

        self.annot1=self.p.annotate("", xy=(0,0), xytext=( 40,-20 ), textcoords="offset points",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="->"))
        self.annot1.set_visible(False)

    def add_curve(self,data,Label):
        ''' pass curve data as (xdata,ydata,format) '''
        self.curves.append( self.p.plot(*data, label=Label, linewidth=3)[0] )
        self.p.legend()

    def _update_annot(self,name, xmouse, ymouse):
        self.annot1.xy = (xmouse, ymouse)
        self.annot1.set_text(name)
        self.annot1.get_bbox_patch().set_alpha(0.4)

    def _hover(self,event):
        vis = self.annot1.get_visible()
        if event.inaxes == self.p:
            _ind = -1
            for i in range(len(self.curves)):
                if (self.curves[i].contains(event)[0]):
                    _ind = i
            if (_ind > -1):
                self._update_annot(self.curves[_ind]._label, event.xdata, event.ydata)
                self.annot1.set_visible(True)
                self.f.canvas.draw_idle()
            elif (vis):
                self.annot1.set_visible(False)
                self.f.canvas.draw_idle()


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
    pops = { i[0]:int(i[1]) for i in np.array(pd.read_csv(f_pop,header=None)) } # pop_data as dict

    # load locations and minus-locations, remove minus, remove locations not affected
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
    pops = { i[0]:int(i[1]) for i in np.array(pd.read_csv(f_pop,header=None)) }

    # combine rows where country is listed multiple times (US, china, etc)
    numbers=np.zeros((len(locs_affected),len(dcon[0,4:]))).astype(int)
    for irow in dcon:
        ind = np.where(irow[1]==locs_affected)
        numbers[ind]+=irow[4:].astype(int)
    # numbers=numbers.astype(int) # possibly needless, commenting out

    # put entities into dictionary
    entity = dict()
    plotlist=np.array(LOCS)
    for i,iname in enumerate(plotlist):
        ind = np.where(iname==locs_affected)[0][0]
        entity[iname]=Country(iname,days,numbers[ind],pops[iname],
                            thresh=args.thresh,smooth=args.smooth)

    # order the list in ascending abs
    vals = np.array([entity[iname].vals.max() for iname in plotlist])
    mask=np.argsort(vals)
    plotlist=plotlist[mask[::-1]]
    vals=vals[mask[::-1]]
    # filter out for top / bottom N countries
    if(args.topAbs!=None):
        plotlist=plotlist[:args.topAbs]
    elif(args.botAbs!=None): plotlist = plotlist[::-1][:args.botAbs]

    # at this point, assign format colors
    for i,iloc in enumerate(plotlist):
        entity[iloc].fmtplot = formats[i]
        print('{}: {}'.format( iloc,entity[iloc].fmtplot ))

    # plot everything
    print('number of affected countries / sovereignties:',len(locs_affected))
    print('plotting {} countries...'.format(len(plotlist)))


    po1=PlotObject('Total {} Cases'.format(args.src),
                   'Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh),
                   'Total Cases [Count]')
    for j,iloc in enumerate(plotlist):
        po1.add_curve(entity[iloc].D_dateAdjusted,
                      '{}({})'.format(iloc,entity[iloc].vals.max()))


    po2=PlotObject('Total {} Cases'.format(args.src),
                   'Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh),
                   'Total Cases [Count]')
    for j,iloc in enumerate(plotlist):
        po2.add_curve(entity[iloc].D_dateAdjNewDaily,
                      '{}({})'.format(iloc,entity[iloc].vals.max()))



    # f3,p3=plt.subplots()
    # wide=1.0
    # width = wide/3
    # for i,iname in enumerate(plotlist):
    #     ydiff = entity[iname]._valsAdj[1:]-entity[iname]._valsAdj[:-1]
    #     p3.bar(entity[iname]._daysAdj[:-1]+wide/len(plotlist)*(i),ydiff,width,label=iname)
    # p3.legend()
    # p3.grid()
    # show plots
    plt.show()
    print('done')
