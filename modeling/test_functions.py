from unittest import TestCase
from modeling.functions import linear_splines_basis1, linear_splines_basis2, ks_calculate, categorical_to_design
from modeling.functions import smooth_linear_splines, decile_plot
from modeling.glm import glm
import pandas as pd
import numpy as np


class TestFunctions(TestCase):
    def test_hats(self):
        """Conducts unit tests for the linear_splines_basis1 function"""
        # set up test data
        n = 1000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 2)), columns=['x1', 'x2'])
        
        # check column names
        knots = np.array([0, .1, .2, .5])
        hts = linear_splines_basis1(df['x1'], knots, 0)
        self.assertEqual(hts.shape[1], 3, 'hats: wrong number of columns returned')
        chknames = hts.columns
        same = (chknames == ['x11', 'x12', 'x13']).sum()
        self.assertEqual(same, 3, 'hats: return colunm names are wrong')
        
        # check sum to one
        hts = linear_splines_basis1(df['x1'], np.array([0, .1, .2, .5]))
        sums = hts.sum(axis=1)
        worst = np.abs(sums - 1)
        worst = worst.max()
        self.assertEqual(worst, 0, 'hats: hats do not sum to one for each obs')
        
        # check <=1
        maxes = hts.max(axis=1)
        maxes = maxes.max()
        self.assertLessEqual(maxes, 1, 'hats: hat bigger than 1 found')
        
        # check >=0
        mins = hts.min(axis=1)
        mins = mins.max()
        self.assertLessEqual(mins, 0, 'hats: hat less than 0 found')
        
        # check x12 by hand (interior hat check)
        x12 = np.zeros(n)
        
        ind = (df.x1 >= knots[1]) & (df.x1 <= knots[2])
        x12a = df.x1[ind]
        wts = (x12a - knots[1]) / (knots[2] - knots[1])
        x12[ind] = wts
        
        ind = (df.x1 >= knots[2]) & (df.x1 <= knots[3])
        x12a = df.x1[ind]
        wts = (x12a - knots[2]) / (knots[3] - knots[2])
        x12[ind] = 1 - wts
        
        diff = np.abs(x12 - hts.x12)
        diff = diff.max()
        self.assertEqual(diff, 0, 'hats: hat calculation wrong--Hat 2')
        
        # check x13 by hand (last hat check)
        x13 = np.zeros(n)
        
        ind = (df.x1 >= knots[2]) & (df.x1 <= knots[3])
        x13a = df.x1[ind]
        wts = (x13a - knots[2]) / (knots[3] - knots[2])
        x13[ind] = wts
        ind = (df.x1 >= knots[3])
        x13[ind] = 1.0
        
        diff = np.abs(x13 - hts.x13)
        diff = diff.max()
        self.assertEqual(diff, 0, 'hats: hat calculation wrong--Hat 3')
    def test_linear_splines2(self):
        n = 1000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 2)), columns=['x1', 'x2'])
    
        # check column names
        knots = np.array([0.1, .15, .2, .5])
        hts = linear_splines_basis2(df['x1'], knots, 0)
        self.assertEqual(hts.shape[1], 5, 'linear_splines_basis2: wrong number of columns returned')
        chknames = hts.columns
        same = (chknames == ['x11', 'x12', 'x13', 'x14', 'x15']).sum()
        self.assertEqual(same, 5, 'linear_splines_basis2: return colunm names are wrong')

        # check x10 is constant
        hts = linear_splines_basis2(df['x1'], knots)
        ones = np.empty(n)
        ones.fill(1)
        diff = hts['x10'] - ones
        chk = diff.var()
        self.assertAlmostEqual(chk, 0, 'linear_splines_basis2: first column is not all 1s')

        # check x11 is x
        diff = hts['x11'] - df['x1']
        chk = diff.var()
        self.assertAlmostEqual(chk, 0, 'linear_splines_basis2: second column is not equal to x')

        # check x12 by hand (interior hat check)
        x12 = np.zeros(n)

        ind = (df.x1 >= knots[0])
        x12[ind] = df['x1'][ind] - knots[0]
        diff = hts['x12'] - x12
        chk = diff.var()
        self.assertAlmostEqual(chk, 0, 'linear_splines_basis2: second column is not equal to x')

    def test_smooth(self):
        from modeling.glm import glm
        def fx(x):
            # very not linear
            #y = -2 * np.sin(2 * 3.14 * x)
            #y[x > 0.75] = 2
            
            #y =1 + 4*x
            
            #y = -2 * np.sin(2 * 3.14 * x)
            #y[x > 0.75] = 2
            y = 1 + 3 * x
            y[x > 0.5] = 2.5

            return y

        n = 30000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 6)), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x 6'])
        y = fx(df.x1) + np.random.normal(0, .5, n)
        df['y'] = y
        hts = np.matrix(linear_splines_basis2(df.x1,np.arange(.05,.95,.05)))
        wts = np.zeros(y.shape[0])
        wts.fill(1)
        xtest = pd.DataFrame(np.arange(0.01,.99,.01),columns=['x'])
        xhts = np.matrix(linear_splines_basis2(xtest.x,np.arange(.05,.95,.05)))

        sm = smooth_linear_splines(hts, y, xhts, 15, wts=wts)
        diff = np.abs(y - np.squeeze(np.array(sm['yhat'])))
        chk = diff.max()
        
        tmp = glm('y~h(x1,(0,.5,1),0)',df,'normal')
        yh = np.squeeze(np.array(tmp.fitted))
        diff = np.abs(np.squeeze(np.array(sm['yhat'])) - yh)
        chk1 = diff.max()
        corr = np.corrcoef(yh, np.squeeze(np.array(sm['yhat'])))
        ok = (corr[0,1] > 0.95)
        
        
        self.assertEqual(ok, True,'smooth_linear_splines: difference exceeds tolerance')

        

    def test_ctd(self):
        """Conducts unit tests for the categorical_to_design function"""
        # set up test data
        n = 1000
        df = pd.DataFrame(np.random.randint(0, 9, (n, 2)), columns=['x1', 'x2'])
        df_out = categorical_to_design(df.x1, 3)['df_out']
        self.assertEqual(df_out.shape[1], 8, 'categorical_to_design: wrong number of columns returned')
        
        # test names
        chk = ['x1:0', 'x1:1', 'x1:2', 'x1:4', 'x1:5', 'x1:6', 'x1:7', 'x1:8']
        ok = (df_out.columns == chk).sum()
        self.assertEqual(ok, 8, 'categorical_to_design: return column names are wrong')
        
        # test values are all binary
        tab = df_out.apply(pd.value_counts)
        chk = (tab.index == [0, 1]).sum()
        self.assertEqual(chk, 2, 'categorical_to_design: entries are not binary')
        
        # check results for x1=2
        zeros = np.zeros(n)
        ones = np.zeros(n)
        ones.fill(1)
        indicator = np.where(df.x1 == 2, ones, zeros)
        chk = (df_out['x1:2'] == indicator).sum()
        self.assertEqual(chk, n, 'categorical_to_design: value for x2 does not match')
        
        # now try with a string input
        a = np.chararray(n, unicode=True)
        a.fill('a')
        b = np.chararray(n, unicode=True)
        b.fill('b')
        c = np.chararray(n, unicode=True)
        c.fill('c')
        ind = np.multiply(df.x1 >= 3, df.x1 <= 5)
        x = np.where(ind, b, a)
        x = np.where(df.x1 > 5, c, x)
        dfx = pd.DataFrame(x, columns=['x'])
        df_out = categorical_to_design(dfx.x)['df_out']
        self.assertEqual(df_out.shape[1], 3, 'categorical_to_design: wrong number of columns returned')
        
        # check names
        chk = (df_out.columns == ['x:a', 'x:b', 'x:c']).sum()
        self.assertEqual(chk, 3, 'categorical_to_design: column names are wrong')
        
        # test values are all binary
        tab = df_out.apply(pd.value_counts)
        chk = (tab.index == [0, 1]).sum()
        self.assertEqual(chk, 2, 'categorical_to_design: entries are not binary')
        
        # check results for x=c
        zeros = np.zeros(n)
        ones = np.zeros(n)
        ones.fill(1)
        indicator = np.where(dfx.x == 'c', ones, zeros)
        chk = (df_out['x:c'] == indicator).sum()
        self.assertEqual(chk, n, 'categorical_to_design: value for x:c does not match')
    
    def test_ks_calculate(self):
        """test ks_calculate"""
        
        # compare Beta(1,1) to Beta(2,1).  The answer is 25%...though we'll check for fun
        n = 10000
        score_points = np.random.uniform(0, 1, n)
        score_points.sort()
        score0 = score_points
        score1 = np.float_power(score_points, 2)
        delta = np.max(np.abs(score0 - score1)) * 100.0
        print('Actual: ' + str(delta))
        
        n = 100000
        score0 = np.random.uniform(0, 1, n)
        score1 = np.random.beta(2, 1, 2 * n)  # make a different length
        score = np.append(score0, score1)
        zeros = np.zeros(n)
        ones = np.zeros(2 * n)
        ones.fill(1)
        binvar = np.append(zeros, ones)
        ks = ks_calculate(score, binvar, plot=True, xlab='This is the x label',ylab='This is the y label',title='A title',subtitle='subtitle')
        print('Estimated: ' + str(ks))
        self.assertAlmostEqual(ks, delta, msg='ks_calculate: incorrect value', delta=.5)

    def test_decile_plot(self):
        n = 30000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 6)), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x 6'])
        df['yr'] = np.random.randint(2000, 2017, n)
        y = -1 + 2 * df['x1'] - 3 * df['x2'] + 1.5 * df['x3'] - .1 * df['x4'] + 7.7 * df['x5']
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
        
        g = glm('y~x1+x2+x3+x4+x5',df,'binomial')
        score = pd.Series(np.squeeze(np.asarray(g.fitted)))
        binvar = pd.Series(np.squeeze(np.asarray(g.y)))
        decile_plot(score, binvar,'The Score','The Actual','Test Title', subtitle='Test Subtitle',correlation=0.01)
        decile_plot(g.fitted, g.y,'The Score','The Actual','Test Title', subtitle='Test Subtitle',correlation=0.01)

