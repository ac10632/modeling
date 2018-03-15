from unittest import TestCase
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import importlib

from modeling.glm import glm


class TestGlm(TestCase):
    def setUp(self):
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
        self.df = df
        
        n = 1500
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
        self.dfnew = df
    
    def test_restrictions(self):
        """
        Tests that the restricions work and work, too, with categorical levels
        """
        r = ['x1=1', 'yr:2001-yr:2002=0', 'yr:2002-yr:2003=0']
        g = glm('ycts~x1 + x2 + x3 + x4 + x5 + x 6+c(yr,2000)', self.df, 'normal',
                restrictions=r)  # test internal space in variable name
        chk = abs(g.t_table.beta['yr:2001'] - g.t_table.beta['yr:2002'])
        chk += abs(g.t_table.beta['yr:2001'] - g.t_table.beta['yr:2003'])
        chk += abs(g.t_table.beta['x1'] - 1)
        self.assertAlmostEqual(chk, 0, 5, 'glm: restrictions')
        
        r = ['x1=1', '2*yr:2001-yr:2002=0', 'yr:2002-0.5*yr:2003=0']
        g = glm('y~x1 + x2 + x3 + x4 + x5 + x 6+c(yr,2000)', self.df, 'binomial',
                restrictions=r)  # test internal space in variable name
        chk = abs(2 * g.t_table.beta['yr:2001'] - g.t_table.beta['yr:2002'])
        chk += abs(g.t_table.beta['yr:2002'] - 0.5 * g.t_table.beta['yr:2003'])
        chk += abs(g.t_table.beta['x1'] - 1)
        self.assertAlmostEqual(chk, 0, 5, 'glm: restrictions')
    
    def test_ks(self):
        from modeling.functions import ks_calculate
        g = glm('y~x1 + x2 + x3 + x4 + x5', self.df, 'binomial')
        ks1 = ks_calculate(g.fitted, g.y, plot=True)  # same function but KS_calculate has been tested
        chk = abs(ks1 - g.ks)
        self.assertAlmostEqual(chk, 0, 0.00001, "glm: KS")
    
    def test_conditionIndex(self):
        self.assertEqual(1, 1)
    
    def test_leverage(self):
        self.assertEqual(1, 1)
    
    def test_fStatistic(self):
        g = glm('ycts~c(yr,2000)+h(x1,(0,.1,.3),0) + x2 + x3 + x4 + x5 + x 6', self.df,
                'normal')  # test internal space in variable name
        f = ((g.total_sum_of_squares - g.sse) / (g.p - 1)) / g.mse
        chk = abs(f - g.model_significance[0])
        self.assertAlmostEqual(chk, 0, 5, 'glm: model_significance')

        g = glm('y~x1 + x2 + x3 + x4 + x5 + x 6', self.df,'binomial')
        mod = smf.logit(formula='y~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        ttt = modfitted.summary()
        l1 =ttt.tables[0][4][3]
        l2 = ttt.tables[0][5][3]
        chisq = 2 * (float(l1.data) - float(l2.data))
        gchisq = g.model_significance[0]
        chk = abs(chisq - gchisq)/chisq
        self.assertAlmostEqual(chk,0,delta=0.02)


    
    def test_rSquare(self):
        """Check R2 calculation by using pandas correlation between y and yhat"""
        g = glm('ycts~x1 + x2 + x3 + x4 + x5 + x 6', self.df, 'normal')  # test internal space in variable name
        df_test = pd.DataFrame(np.append(g.fitted, g.y, axis=1), columns=['y', 'yhat'])
        corr = df_test.corr()
        chk = abs(corr.yhat[0] * corr.yhat[0] - g.r_square)
        self.assertAlmostEqual(chk, 0, 5, 'glm: Rsquare')
    
    def test_totalSumOfSquares(self):
        g = glm('ycts~x1 + x2 + x3 + x4 + x5 + x 6', self.df, 'normal')  # test internal space in variable name
        resid = self.df.ycts - self.df.ycts.mean()
        tss = np.multiply(resid, resid).sum()
        chk = abs(tss - g.total_sum_of_squares)
        self.assertAlmostEqual(chk, 0, 5, "glm: TotalSumOfSquares")
    
    def test_fitted(self):
        """Check fitted values for both logistic and normal regression vs statsmodels"""
        g = glm('y~x1 + x2 + x3 + x4 + x5', self.df, 'binomial')
        mod = smf.logit(formula='y~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        pred = modfitted.predict()
        predglm = g.fitted
        diff = np.array(np.squeeze(np.asarray(predglm)) - pred)
        chk = np.abs(diff).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: fitted values')
        
        g = glm('ycts~x1 + x2 + x3 + x4 + x5', self.df, 'normal')
        mod = smf.glm(formula='ycts~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        pred = modfitted.predict()
        predglm = g.fitted
        diff = np.array(np.squeeze(np.asarray(predglm)) - pred)
        chk = np.abs(diff).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: fitted values')
    
    def test_sse(self):
        """Check sum of squared error vs statsmodels (doesn't apply to logistic regression)"""
        g = glm('ycts~x1 + x2 + x3 + x4 + x5', self.df, 'normal')
        mod = smf.glm(formula='ycts~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        pred = modfitted.predict()
        diff = pred - np.asarray(self.df.ycts)
        chk1 = abs(np.multiply(diff, diff).sum() - g.sse)
        self.assertAlmostEqual(chk1, 0, 5, 'glm: sse calculation')
    
    def test_mse(self):
        self.assertEqual(1, 1)
    
    def test_tTable(self):
        """Check parameters for logistic and normal regression vs statsmodels"""
        g = glm('ycts~x1 + x2 + x3 + x4 + x5', self.df, 'normal')
        mod = smf.glm(formula='ycts~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        diff = np.asarray(modfitted.tvalues) - np.asarray(g.t_table.t)
        chk = np.abs(diff).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: t-statistic calculation')
        
        g = glm('y~x1 + x2 + x3 + x4 + x5', self.df, 'binomial')
        mod = smf.logit(formula='y~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        diff = np.asarray(modfitted.tvalues) - np.asarray(g.t_table.t)
        chk = np.abs(diff).max()
        # not as strict since totally different methods only asymptotically the same
        self.assertAlmostEqual(chk, 0.0, 3, 'glm: t-statistic calculation')
    
    def test_parameters(self):
        
        """Check parameters for logistic and normal regression vs statsmodels"""
        g = glm('ycts~',self.df,'normal')
        chk = float(abs(g.parameters[0] - self.df.ycts.mean()))
        self.assertAlmostEqual(chk, 0, 5, 'glm: mean-only model does not work')
        
        g = glm('ycts~x1 + x2 + x3 + x4 + x5 ', self.df, 'normal')
        mod = smf.glm(formula='ycts~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        diff = np.asarray(modfitted.params) - np.squeeze(g.parameters)
        chk = np.abs(diff).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: parameters calculation')
        
        g = glm('y~x1 + x2 + x3 + x4 + x5', self.df, 'binomial')
        mod = smf.logit(formula='y~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        diff = np.asarray(modfitted.params) - np.squeeze(g.parameters)
        chk = np.abs(diff).max()
        ttt=modfitted.summary()
        self.assertAlmostEqual(chk, 0, 5, 'glm: parameters calculation')
    
    def test_predict(self):
        """Check predicted values for logistic and normal regression vs statsmodels"""
        g = glm('y~x1 + x2 + x3 + x4 + x5', self.df, 'binomial')
        mod = smf.logit(formula='y~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        pred = modfitted.predict(self.dfnew)
        predglm = g.predict(self.dfnew)
        diff = np.squeeze(np.asarray(predglm)) - pred
        chk = np.abs(diff).max()
        
        g = glm('ycts~x1 + x2 + x3 + x4 + x5', self.df, 'normal')
        mod = smf.glm(formula='ycts~x1 + x2 + x3 + x4 + x5', data=self.df)
        modfitted = mod.fit()
        pred = modfitted.predict(self.dfnew)
        predglm = g.predict(self.dfnew)
        diff = np.squeeze(np.asarray(predglm)) - pred
        chk1 = np.abs(diff).max()
        
        self.assertAlmostEqual(chk + chk1, 0, 5, 'glm: predicted values')
    
    def test_implement(self):
        g = glm('ycts~c(yr,2000)+h(x1,(0,.1,.3),0) + l(x2,(0.1,.5,.6)) + x3 + x4 + x5 + x 6', self.df,
                'normal')  # test internal space in variable name
        g.implement('/home/will/PycharmProjects/modeling_tools/temp/modeltest.py')
        import temp.modeltest as m
        from temp.modeltest import model
        yhat = model(self.df)
        tmp = pd.DataFrame(yhat)
        tmp.columns = ['file']
        tmp['internal'] = g.fitted
        chk = abs(tmp.internal - tmp.file).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: Implementation')
        
        g = glm('y~c(yr,2000)+h(x1,(0,.1,.3),0) + x2 + x3 + x4 + x5 + x 6', self.df,
                'binomial')  # test internal space in variable name
        g.implement('/home/will/PycharmProjects/modeling_tools/temp/modeltest.py')
        importlib.reload(m)
        from temp.modeltest import model
        yhat = model(self.df)
        tmp = pd.DataFrame(yhat)
        tmp.columns = ['file']
        tmp['internal'] = g.fitted
        chk = abs(tmp.internal - tmp.file).max()
        self.assertAlmostEqual(chk, 0, 5, 'glm: Implementation')

