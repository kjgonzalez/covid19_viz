'''
main goal: create a graph per-country of population-percentage vs days since
    first infection
NOTE: will start with just number of infections
want:
countr[0]=[
[days,infections],
]

note: over 154 unique countries

quick note about plotting:
7 diff colors: b / g / r / c / m / y / k
4 diff linestyles: - / -- / -. / :
9 diff markers: <none> / , / o / v / ^ / s / * / + / x

sources:https://digitalsynopsis.com/design/color-schemes-palettes/

STAT | DESCRIPTION
???? | plot with specific color scheme
???? | be able to plot top 5 of a certain category
???? | create interpolating color scheme
???? |
???? |
???? |


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, argparse
import klib
from scipy.interpolate import make_interp_spline, BSpline
from matplotlib import cm

p1=np.array([
[0.984,0.643,0.396,1],
[0.973,0.431,0.318,1],
[0.933,0.243,0.220,1],
[0.820,0.098,0.243,1],
])


# first, generate all possible combinations, with priority to color & marker
formats = []
for ils in '- -- -. :'.split(' '):
    for imk in ' , o * v x ^ s +'.split(' '):
        for iclr in 'bgrcmyk':
            formats.append( iclr+ils+imk )


class Country:
    def __init__(self,name,days,values,pop,thresh=0,smooth=False):
        self.name=name
        self.days=days
        self.vals=values
        self.pop=pop
        self._thresh=thresh

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

def dayssince(dates,fmt='%m/%d/%Y'):
    seconds = np.array([time.mktime(time.strptime(i+'20',fmt)) for i in dates])
    days = np.round(seconds/(60*60*24))
    return days

base='COVID-19\\csse_covid_19_data\\csse_covid_19_time_series\\time_series_19-covid-{}.csv'
f_con=base.format('Confirmed')
f_dea=base.format('Deaths')
f_rec=base.format('Recovered')
f_pop='pops.csv'
eu = 'Austria Belgium Bulgaria Croatia Cyprus Czechia ' + \
    'Denmark Estonia Finland France Germany Greece Hungary Ireland Italy '+\
    'Latvia Lithuania Luxembourg Malta Netherlands Poland Portugal Romania '+\
    'Slovakia Slovenia Spain Sweden'
eu=eu.split(' ')


if(__name__=='__main__'):
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--src',default='Confirmed',help='Confirmed, Deaths, Recovered')
    p.add_argument('--thresh',default=50,type=int,help='min number of cases for aligning chronologically')
    p.add_argument('--locs',help='desired countries. example: China-US-Sri_Lanka',
        default='China-US-Italy-Germany')
    p.add_argument('--smooth',default=False,action='store_true',help='create smooth curves')
    p.add_argument('--allPlots',default=False,action='store_true',help='plot all plots')
    p.add_argument('--topAbs',default=None,type=int,help='top N countries by absolute number')
    args=p.parse_args()


    # load data
    pdcon = pd.read_csv(base.format(args.src))
    dates=np.array(pdcon.columns[4:])
    fmt='%m/%d/%Y'
    fmt2='%Y%b%d'
    date0=time.strftime(fmt2,time.strptime(dates[0]+'20',fmt))
    dcon=np.array(pdcon)
    days = dayssince(dates)
    days = days-days[0]
    names = klib.listContents(dcon[:,1],True)[:,0] # names of countries
    names = names[np.argsort(names)] # sort names alphabetically
    pdpops = np.array(pd.read_csv(f_pop,header=None))
    pops = { i[0]:int(i[1]) for i in pdpops }

    # preprocess locations, including regional blocks
    if(args.locs=='eu'):
        args.locs=eu
    elif(args.locs=='all'):
        args.locs=names
    else:
        args.locs = args.locs.replace('_',' ').split('-')

    print('number of available plot formats:',len(formats))
    print('number of affected countries / sovereignties:',len(names))
    print('plotting {} countries...'.format(len(args.locs)))
    # check that all values in args.locs are in pops
    for iloc in args.locs:
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
    numbers=np.zeros((len(names),len(dcon[0,4:]))).astype(int)
    for irow in dcon:
        ind = np.where(irow[1]==names)
        numbers[ind]+=irow[4:].astype(int)
    numbers=numbers.astype(int)

    # put all data into dict of objects
    data = dict()
    plotlist=np.array(args.locs)
    for iname in plotlist:
        ind = np.where(iname==names)[0][0]
        data[iname]=Country(iname,days,numbers[ind],pops[iname],
                        thresh=args.thresh,smooth=args.smooth)
    # order the list in ascending abs
    vals = np.array([data[iname].vals.max() for iname in plotlist])
    mask=np.argsort(vals)
    plotlist=plotlist[mask[::-1]]
    vals=vals[mask[::-1]]
    for i in range(len(vals)):
        print(plotlist[i],vals[i])
    # plot only main one
    if(args.topAbs!=None):
        plotlist=plotlist[:args.topAbs]



    if(args.allPlots):
        # plot everything
        f,p = plt.subplots(2,2)
        p=np.reshape(p,4)
        f.suptitle(args.src + ' infection cases')
        for j,i in enumerate(plotlist):
            p[0].plot(data[i].days,data[i].vals+1,formats[j],label='{} ({:e})'.format(i,pops[i]))
            # note: adding +1 for compatabilitty with log plotting
        p[0].set_ylabel('Number of Cases [n]')
        p[0].set_xlabel('Days since world outbreak ('+date0+') [Days]')
        p[0].grid()
        # p[0].set_yscale('log')

        # p[1].legend(loc=2,fontsize='x-small',ncol=2)
        indPlot=0
        if(len(plotlist)<20):
            p[indPlot].legend(loc=2)
        elif(len(plotlist)>20 and len(plotlist) <50):
            p[0].legend(loc=2,fontsize='x-small',ncol=2)
        else:
            p[0].legend(loc=2,fontsize=4,ncol=3)
        for j,i in enumerate(plotlist):
            p[1].plot(data[i].days,data[i].percentpop,formats[j],label='{} ({:e})'.format(i,pops[i]))
        p[1].set_ylabel('Population Percentage [%]')
        p[1].set_xlabel('Days since world outbreak ('+date0+') [Days]')
        p[1].grid()


        for j,i in enumerate(plotlist):
            p[2].plot(data[i].daysAdj,data[i].valsAdj,formats[j],label='{} ({:e})'.format(i,pops[i]))
        p[2].set_ylabel('Number of Cases [n]')
        p[2].set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
        p[2].grid()

        for j,i in enumerate(plotlist):
            p[3].plot(data[i].daysAdj,data[i].percentpopAdj,formats[j],label='{} ({:e})'.format(i,pops[i]))
        p[3].set_ylabel('Population Percentage [%]')
        p[3].set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
        p[3].grid()
        plt.show()
    else:
        palette = cm.get_cmap('plasma',len(plotlist)+2)
        colors = palette.colors[:-2][::-1]

        f,p = plt.subplots(figsize=(10,7))
        for j,iloc in enumerate(plotlist):
            p.plot(data[iloc].daysAdj, data[iloc].per100kAdj, label='{} ({:.3}m)'.format(iloc, pops[iloc]/1e6),linewidth=3)
        p.set_ylabel('Cases/100,000 [Count]')
        p.set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
        f.suptitle('''{} Cases per 100,000 people
        Adjusted for start of outbreak (thresh:{})
        10 Most Affected EU Member Countries'''.format(
            args.src,args.thresh,args.topAbs
        ))
        p.grid()
        p.legend()

        f2, p2 = plt.subplots(figsize=(10,7))
        for j, iloc in enumerate(plotlist):
            p2.plot(data[iloc].daysAdj, data[iloc].valsAdj,
                    label='{} ({:.3}m)'.format(iloc, pops[iloc] / 1e6),
                    linewidth=3)
        p2.set_ylabel('Total Cases [Count]')
        p2.set_xlabel('Days since per-country outbreak, adjusted [Days] (thresh:{})'.format(args.thresh))
        f2.suptitle('''Total {} Cases
        Adjusted for start of outbreak (thresh:{})
        Placeholder'''.format(
            args.src, args.thresh, args.topAbs
        ))
        p2.grid()
        p2.legend()
    # show plots
    plt.show()
    print('done')