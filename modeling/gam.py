from modeling.glm import glm as glm
from modeling.functions import linear_splines_basis2, smooth_linear_splines
from modeling.data_class import ModelError
import numpy as np
import pandas as pd




class gam(glm):
    """
    Fits a Generalized Additive Model (gam).  gam inherits from glm.
    
    gam offers no methods or attributes not available in glm.  What it offers is an additional type of term
    within the model *formula*.
    
    The new type of term specifies that the effect is to be fit using a flexible functional
    form--smoothing splines.  The splines used by gam are :meth:`~functions.linear_splines_basis2`.
    
    The syntax for each gam term in the *formula* is
    
        *s* (*x*)
        
        or
        
        *s* (*x*, *k*, *num_turn*)
        
    - *x*. *x* is the name of the variable from the input DataFrame.
    
    - *k*.   *k* is a value from 3 to 100 that controls the number of knot points.
      A value of *k* produces *k* knot points.  The points are based on the quantiles of *x*. The default
      value of *k* is 10 and *num_turn* is 5.
      
    - *num_turn*. *num_turn* is the maximum number of turns (maxima/minima) permitted in the smoothed function.
      The default value of *num_turn* is 5.
    
    The smoothing spline algorithm is characterized as ridge regression where the ridge parameter is applied to
    all terms (basis functions) except the intercept and global linear term.  Each term receives its own ridge
    parameter. The algorithm estimates the ridge parameters via a random holdout.

    The estimation approach is:
    
    - Estimate the ridge parameters using the backfitting algorithm.
    
    - Re-estimate the model as a glm model using the ridge parameters that come out of the backfit.  Since, given the
      ridge parameters, the splines are linear in parameters the model reduces to a glm model.
      
    - For the Binomial family, the backfitting algorithm is nested within the IRWLS algorithm.
    
    Note: if there are no gam terms, it is more efficient to use glm.

    For the smoothing spline algorithm, see
    
        https://www.whitman.edu/Documents/Academics/Mathematics/Griggs.pdf section 6.1
        
    
    For the backfitting algorithm, see
    
        http://www.d.umn.edu/math/Technical%20Reports/Technical%20Reports%202007-/TR%202007-2008/TR_2008_8.pdf

    """
    def __init__(self, formula, df, family, wts=None, event=None, restrictions=None):
        # pandas DataFrame of input data
        self.df = df
        self.__formula = ''
        #self._DataClass.__wts = None
        # dictionary of spline design matrices
        self.__sxs = None
        # dictionary of test points
        self.__sxtest = None
        # dictionary of knot points
        self.__sus = None
        # number of turns for each gam term
        self.__snum_turn = None
        # glm portion of the formula
        self.__glm_formula_partial = None
        # gam portion of the formula re-cast as a linear spline (basis2) specification
        self.__gam_formula = None
        # array of ridge parameters as returned from backfit
        self.__ridge_parameters = None
        # array of names of effects specified as gam (used in s( )).
        self.__gam_vars = []
        
        # this pulls apart the glm/gam portions of the formula and builds the design matrices for the gam factors.
        self.__reparse(formula)
        # full model formula specified as a glm model
        global_formula = self.__glm_formula + '+' + self.__gam_formula

        self.family = family

        # the first step is to get the design matrix, y vector for the glm portion of the spec
        glm.__init__(self, self.__glm_formula, df, family, fit=False)
        y = self.y.copy()
        if self.family == 'BINOMIAL':
            bwts = np.zeros(y.shape[0])
            bwts.fill(1)
        else:
            y = self.y.copy()
            bwts = wts
        self.__fit_gam(y,bwts)
        # now build the ridge vector for the 'full' glm model--the straight glm portion with the gam portion recast
        # as linear splines.
        ridges = np.zeros(self.p)
        for j in range(len(self.__gam_vars)):
            p = self.__sxs[self.__gam_vars[j]].shape[1]
            r = self.__ridge_parameters[j]
            this_ridge = np.zeros(p-2)
            this_ridge.fill(r)
            ridges = np.append(ridges, np.zeros(1))
            ridges = np.append(ridges, this_ridge)
        glm.__init__(self, global_formula, df, family, event=event, ridge=ridges, wts=wts, restrictions=restrictions)
        
    def __fit_gam(self, y, wts):
        """
        This routine conducts the backfit to estimate the gam model.  The output of the routine is the ridge
        parameter of each gam effect.
        
        :return: self.__ridge_parameters.  Ridge parameter for each gam effect.
        :rtype: numpy array
        """
        
        # this is for the glm portion of the model
        if (self.family == 'NORMAL') or (self.family == 'BINOMIAL'):
            x = self.x.copy()
            y = y.copy()
            if wts is not None:
                sqrtwts = np.matrix(np.sqrt(wts)).T
                x = np.multiply(x, sqrtwts)
                y = np.multiply(y, sqrtwts)
            xpxi = (x.T * x).I
            # gam effect variable names
            var_names = self.__gam_vars # ['x1','x4'] #list(self.__sxs.keys())
            # initialize ridge vector
            self.__ridge_parameters = np.zeros(len(var_names))
            zeros = np.matrix(np.zeros(self.x.shape[0])).T
            oldyhat = None
            # holds the model output for each effect. Each gam effect gets its own column but all glm effects are
            # combined (column 0)
            yhats = np.matrix(np.zeros(self.x.shape[0] * (len(var_names) + 1)).reshape(self.x.shape[0], 1+len(var_names)))
            for j in range(13):
                yhats[:, 0] = zeros
                # glm portion
                yres = y - yhats.sum(1)
                if wts is not None:
                    xpy = x.T * np.multiply(yres, sqrtwts)
                else:
                    xpy = x.T * yres
                beta = xpxi * xpy
                yhat = self.x * beta
                yhats[:, 0] = yhat
                # cycle through gam effects
                for (col, var_name) in enumerate(var_names):
                    yhats[:, col+1] = zeros
                    yres = y - yhats.sum(1)
                    amean = yres.mean()
                    amax = yres.max()
                    amin = yres.min()
                    sm = smooth_linear_splines(self.__sxs[var_name], yres,self.__sxtest[var_name], self.__snum_turn[var_name], wts=wts)
                    atmp = sm['ridge']
                    self.__ridge_parameters[col] = sm['ridge']
                    yhats[:, col+1] = sm['yhat']
                yhat = yhats.sum(1)
                if oldyhat is not None:
                    delta = oldyhat - yhat
                    ssnum = float(delta.T * delta)
                    ssden = float(oldyhat.T * oldyhat)
                    criterion = ssnum / ssden
                    if criterion < 0.00001:
                        break
                oldyhat = yhat
                if self.family == 'BINOMIAL':
                    yhat_exp = np.exp(yhat)
                    p0 = yhat_exp / (1 + yhat_exp)
                    wt = np.squeeze(np.asarray(p0 / (1 + yhat_exp)))
                    y = np.reshape(np.squeeze(np.asarray(self.y)) / wt,
                                    (self.n, 1)) - 1 - yhat_exp + yhat
                if j == 13:
                    raise ModelError('gam: iterations exceeded')


    
    def __reparse(self, formula):
        """
        This routine
        
            - strips out the glm portion of the request from the user
            - creates the knot points for each spline variable
            - creates the design matrix for each spline variable
            
        
        :param formula: model formula as specified by user
        :type formula: str
        :return: self.__glm_formula_partial: glm portion of the user-specified formula
        :rtype: str
        :return: self.__gam_formula: full model with gam components converted to linear_splines_basis2
        :rtype: str
        :return: self.__gam_vars: names of gam effects
        :rtype: list
        :return: self.__sxs: design matrices of splines
        :rtype: dict
        :return: self.__sxtest: design matrices of splines at smoothness test points
        :return: self.__sus: knot points for splines
        :rtype: dict
        :return: self.__snum_turn: # of turns permitted
        :rtype: dict
        """
        if formula is None:
            raise ModelError('gam: no formula')
        # see if there is a dependent variable (this is optional...predict method doesn't need one
        i = formula.find('~')
        if i <= 0:
            raise ModelError('gam: no dependent variable')
        left_right = formula.split('~')
        r = left_right[1].split('+')
        glm_formula = None
        gam_formula = None
        for factor in r:
            factor = factor.strip()
            if factor.find('s(') < 0:
                if glm_formula is None:
                    glm_formula = factor
                else:
                    glm_formula += ' + ' + factor
            else:
                f1 = factor[2:len(factor)]  # drop 's('
                # Insert here the smoothing option
                icomma = f1.find(',')
                iparen = f1.find(')')  # find variable name
                if icomma >= 0:
                    var_name = f1[0:icomma]
                    remainder = f1[(icomma+1):iparen]
                    icomma = remainder.find(',')
                    smooth_param_str = remainder[0:icomma]
                    num_turn_str = remainder[(icomma+1):iparen]
                    try:
                        smooth_param = float(smooth_param_str)
                    except ValueError:
                        raise ModelError('gam: smoothing parameter must be a number')
                    if (smooth_param < 3.0) or (smooth_param > 100.0):
                        raise ModelError('gam: smoothing parameter is >= and <=100')
                    try:
                        num_turn = float(num_turn_str)
                    except ValueError:
                        raise ModelError('gam: number of turns must be a number')
                    if (num_turn < 2.0) or (num_turn > smooth_param):
                        raise ModelError('gam: number of turns must be >=2 and <= # knots')
                else:
                    var_name = f1[0:iparen]
                    smooth_param = 10
                    num_turn = 5
                self.__gam_vars += [var_name]
                if (self.df.columns == var_name).sum() != 1:
                    raise ModelError('gam: factor ' + var_name + ' not in DataFrame')
                num_knots = smooth_param
                num_knots_r = 1 / num_knots
                u = np.arange(num_knots_r, 1.00001 - num_knots_r, num_knots_r)
                # unique: don't want duplicates which could happen if the data is lumpy
                knots = np.unique(np.asarray(self.df[var_name].quantile(u)))
                # make sure the knot points are not at the min or max
                if (self.df[var_name] < knots[0]).sum() < 0.02*self.df.shape[0]:
                    knots = knots[range(1,knots.size)]
                if (self.df[var_name] < knots[knots.size-1]).sum() < 0.02*self.df.shape[0]:
                    knots = knots[range(0,knots.size-1)]
                splines = linear_splines_basis2(self.df[var_name], knots)
                mn = self.df[var_name].quantile(0.01)
                mx = self.df[var_name].quantile(0.99)
                delta = (mx - mn)/100
                xtest = pd.DataFrame(np.arange(mn,mx,delta),columns=['x'])
                xtest_splines = linear_splines_basis2(xtest.x, knots)
                # make into glm formula
                knots_str = '('
                for (j, k) in enumerate(knots):
                    knots_str += str(k)
                    if j < knots.size - 1:
                        knots_str += ','
                knots_str += ')'
                if self.__sxs is None:
                    self.__sus = {var_name: knots}
                    self.__sxs = {var_name: np.matrix(splines)}
                    self.__sxtest = {var_name: np.matrix(xtest_splines)}
                    self.__snum_turn = {var_name: num_turn}
                    gam_formula = 'l(' + var_name + ',' + knots_str + ')'
                else:
                    self.__sxs[var_name] = np.matrix(splines)
                    self.__sxtest[var_name] = np.matrix(xtest_splines)
                    self.__sus[var_name] = knots
                    self.__snum_turn[var_name] = num_turn
                    gam_formula += '+l(' + var_name + ',' + knots_str + ')'
        if glm_formula is None:
            glm_formula = left_right[0]
        else:
            glm_formula = left_right[0] + ' ~ ' + glm_formula
        self.__glm_formula = glm_formula
        self.__gam_formula = gam_formula





