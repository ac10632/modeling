from unittest import TestCase
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from modeling.gam import gam

#TODO: test restrictions, event, What else?

class TestGam(TestCase):
    def setUp(self):
        n = 300000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 6)), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x 6'])
        # df.x2 = .6*df.x1 + .4*df.x2
        # df.x3 = .6*df.x1 + .4*df.x3
        df['yr'] = np.random.randint(2000, 2017, n)
        #y = fx(df.x1, df.x4, df.x3, df.x2)  # -1 + 2 * np.sin(df.x1*6.28)
        z = 3 * df.x1
        z[df.x1 > .5] = 1.5  # 10*(df.x1-.25)**2*(df.x1-.75)**2
        y = -4 + 2 * df.x3 + 4 * df.x2 + z + 10 * (df.x4 - .5) ** 2  # 10*np.sin(6.28*df.x4)
        # y[df.x1>0.75] = -3
        # + 1.5 * df['x3']
        p = np.exp(y) / (1 + np.exp(y))
        df['p'] = p
        df.p.describe()
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        y1.fill(1.0)
        u = np.random.uniform(0, 1, n)
        yz = np.where(u < p, y1, y0)
        df['y'] = yz
        y0 = np.chararray(n)
        y0.fill('n')
        y1 = np.chararray(n)
        y1.fill('y')
        yc = np.where(u < p, y1, y0)
        df['yc'] = yc
        df['ycts'] = y + np.random.normal(0, 1, n)
        df['ym'] = y
        self.df = df
        self.p = p

    def test_gam_normal(self):
        tmp = gam('ycts~x2+x3+s(x1,20,5)+s(x4,20,5)', self.df, 'normal')  # +s(x4,10)
        #plt.plot(self.df.ym, tmp.fitted, 'o')
        #plt.waitforbuttonpress()
        #plt.close()
        #print(tmp.t_table[0:4])
        #tmp.plot_effects()
        #tmp.plot_effects('x4')
        #tmp.plot_effects('x1')
        
        effect='x4'
        qs = self.df[effect].quantile([0.01, .99])
        x_values = np.arange(qs[0.01], qs[0.99], (qs[0.99] - qs[0.01]) / 100)
        knots = tmp.factors[effect]['knots']
        z = tmp.parameters.index
        y_values_x = tmp.parameters[[(lambda x: x.find(effect) >= 0)(x) for x in z]]
        yh = np.zeros(x_values.size)
        omit = tmp.factors[effect]['omit']
        if (omit is None) or (omit == 0):
            y_values = y_values_x
            yh = y_values[0] * x_values
        else:
            y_values = np.zeros(knots.size)
            y_values[np.arange(x_values.size) != omit] = y_values_x
        for (j, knot) in enumerate(knots):
            i = x_values >= knot
            yh[i] += (x_values[i] - knot) * y_values[j + 1]
        y_actual = 10*(x_values - 0.5)**2
        y_actual -= y_actual.mean()
        yh -= yh.mean()
        delta = abs(y_actual - yh)
        chk = delta.max()
        self.assertAlmostEqual(chk,0,msg='gam: normal family too different',delta=0.1)
        #plt.close()
        #plt.plot(x_values, delta)
        #plt.waitforbuttonpress()
        #plt.close()

    def test_gam_binomial(self):
        tmp = gam('y~x2+x3+s(x1,10,3)+s(x4,10,5)', self.df, 'binomial')  # +s(x4,10)
        r = tmp.ridge
        # plt.plot(self.df.ym, tmp.fitted, 'o')
        # plt.waitforbuttonpress()
        # plt.close()
        # print(tmp.t_table[0:4])
        # tmp.plot_effects()
        # tmp.plot_effects('x4')
        # tmp.plot_effects('x1')
    
        effect = 'x4'
        qs = self.df[effect].quantile([0.01, .99])
        x_values = np.arange(qs[0.01], qs[0.99], (qs[0.99] - qs[0.01]) / 100)
        knots = tmp.factors[effect]['knots']
        z = tmp.parameters.index
        y_values_x = tmp.parameters[[(lambda x: x.find(effect) >= 0)(x) for x in z]]
        yh = np.zeros(x_values.size)
        omit = tmp.factors[effect]['omit']
        if (omit is None) or (omit == 0):
            y_values = y_values_x
            yh = y_values[0] * x_values
        else:
            y_values = np.zeros(knots.size)
            y_values[np.arange(x_values.size) != omit] = y_values_x
        for (j, knot) in enumerate(knots):
            i = x_values >= knot
            yh[i] += (x_values[i] - knot) * y_values[j + 1]
        y_actual = 10 * (x_values - 0.5) ** 2
        y_actual -= y_actual.mean()
        yh -= yh.mean()
        delta = abs(y_actual - yh)
        chk = delta.max()
        print(self.df.y.mean())
        if chk > 0.4:
            plt.plot(x_values,y_actual,'red')
            plt.plot(x_values,yh,'black')
            plt.waitforbuttonpress()
            plt.close()
            tmp.plot_effects()
            print(tmp.t_table)
            
        print(chk)
        print(r)
        self.assertAlmostEqual(chk, 0, msg='gam: normal family too different', delta=0.4)
        # plt.close()
        # plt.plot(x_values, delta)
        # plt.waitforbuttonpress()
        # plt.close()

    def test_real(self):
        from modeling.gam import gam
        #dfx = pd.read_csv('/home/will/Testing/CSVData/MortgageDataRandom.csv')
        #i = dfx.EndingDQ >= 0
        #dfx1 = dfx[i].copy()
        #dfx1.index = np.arange(0,dfx1.shape[0])
        #default = np.zeros(dfx1.shape[0])
        #default[dfx1.EndingDQ >= 6] = 1
        #dfx1['default'] = default
        #dfx1['ltv'] = dfx1.UPB*100/dfx1.PValue
        #dfx1.to_csv('~/Testing/CSVData/test.csv')
        dfx1 = pd.read_csv('~/Testing/CSVData/test.csv')
        print(dfx1.default.mean())
        #
        # put floor under ridge and test results here...say 8.
        dfx1 = dfx1[dfx1.ltv < 120]
        dfx1 = dfx1[dfx1.PValue < 1000000]
        dfx1.PValue = dfx1.PValue/1000
        dfx1.index = np.arange(0,dfx1.shape[0])
        g = gam('default ~  FICO + s(ltv,20,5) + s(PValue,20,5)+s(PayRatio,10,5)',dfx1,'binomial')
        b = g.t_table
        print(b)
        g.plot_effects()
        r=1