import math
import numpy as np
import pandas as pd
import numpy.linalg as linalg
import scipy.stats as stats
import matplotlib.pyplot as plt
from modeling.data_class import DataClass, ModelError


class glm(DataClass):
    
    """
    This class fits a Generalized Linear Model.  This class inherits from DataClass.

    The call is:

    Example Usage:
    ::

      glm(formula, df, family, wts = None, event = None, restriction = None, ridge=None, fit=True)


    - df. The pandas DataFrame with the data to use in the model fit.

    These parameters are documented separately:

    - formula. The model to fit.
    - family. The regression family dictating the model type.
    - wts. Optional regression weights (Normal family only).
    - event. Optional specification of the event to model for logistic regression.
    - restrictions. Optional restrictions on the coefficients.
    - ridge. Optional ridge parameters for ridge regression.
    
    Finally,
    
    - fit. Optional.  If False, the model is not fit.  Mostly for use by gam.

    Example:
    ::
    
        g = glm('default ~ credit + h(ltv,(40,70,90,130),0)', myData, family='binomial')

    **Methods**

    - :meth:`~glm.implement`

    - :meth:`~glm.plot_effects`

    - :meth:`~glm.predict`
    
    **Attributes**
    
    - :meth:`~glm.condition_index`

    - :meth:`~data_class.DataClass.event`

    - :meth:`~data_class.DataClass.factors`
    
    - :meth:`~data_class.DataClass.family`

    - :meth:`~glm.fitted`

    - :meth:`~data_class.DataClass.formula`

    - :meth:`~data_class.DataClass.has_intercept`

    - :meth:`~data_class.DataClass.implementation_skeleton`

    - :meth:`~glm.ks`

    - :meth:`~glm.leverage`

    - :meth:`~data_class.DataClass.model_degrees_freedom`

    - :meth:`~glm.model_significance`

    - :meth:`~glm.mse`

    - :meth:`~data_class.DataClass.n`

    - :meth:`~data_class.DataClass.p`

    - :meth:`~glm.parameters`

    - :meth:`~glm.r_square`

    - :meth:`~glm.restrictions`

    - :meth:`~glm.ridge`

    - :meth:`~glm.sse`

    - :meth:`~glm.t_table`

    - :meth:`~glm.total_sum_of_squares`

    - :meth:`~data_class.DataClass.wts`

    - :meth:`~data_class.DataClass.x`

    - :meth:`~data_class.DataClass.x_column_names`

    - :meth:`~data_class.DataClass.y`



    """
    
    def __init__(self, formula, df, family, wts = None, event = None, restrictions = None, ridge = None, fit = True):
        
        DataClass.__init__(self, df)
        self.family = family
        self.formula = formula
        if wts is not None:
            if self.family == 'BINOMIAL':
                raise ModelError('glm: cannot specify weights with family BINOMIAL')
            self.wts = wts
        self.event = event
        
        self.build_design()
        # x.T * x for the design matrix
        self.__xpx = None
        # inverse of the xpx
        self.__xpxI = None
        # estimated parameters: pandas Series
        self.__parameters = None
        # table of parameters, standard errors, t-statistics, p-values: pandas DataFrame
        self.__ttable = None
        # mean squared error (for family normal): float
        self.__mse = None
        # sum of squared error (for family normal): float
        self.__sse = None
        # total sum of squares (for family normal): float
        self.__tss = None
        # fitted values (yhats) : numpy vector
        self.__fitted = None
        # F statistic: float
        self.__Fstatistic = None
        # restrictions on the parameters as specified by the user
        self.__restrictions = restrictions
        # matrix of restriction coefficients (LHS of restriction equation): numpy matrix
        self.__restriction_coefficients = None
        # vector of restriction values (RHS of restriction equation): numpy array
        self.__restriction_values = None
        # diagonal of the hat matrix
        self.__hat_diagonal = None
        # KS statistic (for family binomial): float
        self.__KS = None
        # final regression weights from IRWLS (form family binomial): numpy vector
        self.__chisq_statistic = None
        self.__f_statistic = None
        if restrictions is not None:
            self.__build_restriction_matrices(restrictions)
        # must be after build_design bc need to know # of columns in self.x
        self.__ridge = self.__check_ridge(ridge)
        if fit:
            self.__glm_fitter()
            if self.ridge is not None:
                self.__ttable['ridge'] = self.ridge

    @property
    def ridge(self):
        """
        Parameters for ridge regression.  Size of array is *p*.  The values correspond  to the effects in the
        model in the same order as they appear in the user formula, which is the same order as the attribute
        t_table. If ridge has been specified, the values are returned as part of t_table.
        
        :return: ridge parameters specified by user
        :rtype:  numpy array
        
        
        """
        return self.__ridge
    
    def __check_ridge(self, value):
        """
        Checks that the user-supplied ridge values are OK.
        :param value: ridge values for ridge regression
        :type value: numpy array, vector or pandas Series
        :return: ridge values
        :rtype: numpy column vector
        
        
        """
        if value is None:
            self.__ridge = None
            return
        if str(type(value)).find('numpy.ndarray') >= 0:
            value = np.matrix(value).T
        else:
            if str(type(value)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
                if value.shape[0] == 1:
                    value = value.T
            else:
                raise ModelError('glm: ridge must be numpy array or vector')
        if value.shape[0] != self.p:
            raise ModelError('glm: ridge array must have p entries')
        return value

    @property
    def restrictions(self):
        """
        The user may specify linear restrictions on the coefficients.

        Each restriction is specified as a str.  The format is:

            '*a1* * *x1* + *a2* * *x2* + ... + *xk* * *xk* = *c*'

        where

            - *a1*, *a2* are constants

            - *x1*, *x2* are names of parameters in the model

            - *c* is a constant

        Multiple restrictions are put into a list.  If the restrictions refer to effects that are derived--that is
        use *c()*, *h()* or *l()* syntax, the effect name is the derived name as shown in t_table or parameters
        attributes.

        Example Usage
         ::

           r = ['3*temperature - 4.2*pressure = 0','meaning = 42']


        :return: user-specified restrictions on the coefficients
        :rtype: list of str


        """
        return self.__restrictions
    
    @property
    def ks(self):
        """
        Kolmogorov-Smirnov statistic.

        Note. Only applies to family Binomial.

        :return: KS statistic
        :rtype: float


        """
        return self.__KS
    
    @property
    def condition_index(self):
        """
        Condition index.  This is a measure of collinearity within the design matrix.

        For logistic regression, this is based on the last iteration of the IRWLS algorithm.
        The condition index is

        sqrt(max eigenvalue/min eigenvalue) of X'X

        Values above 30 are a concern.

        :return: condition index
        :rtype: float


        """
        
        eig = linalg.eigvals(self.__xpx)
        return math.sqrt(max(eig) / min(eig))
    
    @property
    def leverage(self):
        """
        Leverage points.

        For logistic regression, this is based on the last iteration of the IRWLS algorithm.  This is available
        only if *n* <= 1000.

        :return: row numbers in *df* that are deemed to be of higher leverage.
        :rtype: numpy array


        """
        if self.__hat_diagonal is None:
            return None
        big = self.__hat_diagonal > 2.0 * float(self.n) / float(self.p)
        if big.any():
            return np.arange(self.n)[big]
        else:
            return None
    
    @property
    def model_significance(self):
        """
        Model significance statistic.  The return depends on the family:
        
        - Normal family. The return is the F-statistic.
        - Binomial family. The return is the chi-square statistic based on the likelihood ratio.

        The statistics are given only if the model has an intercept.

        :return: statistic, degrees of freedom, p-value.
        :rtype: tuple


        """

        if self.has_intercept:
            num_df = float(self.p - 1)
            if  (self.family == 'NORMAL'):
                den_df = float(self.model_degrees_freedom)
                return self.__f_statistic, num_df, den_df, 1 - stats.f.cdf(self.__f_statistic, num_df, den_df)
            else:
                return self.__chisq_statistic, num_df, 1 - stats.chi2.cdf(self.__chisq_statistic, num_df)
        else:
            return None
    
    @property
    def r_square(self):
        """
        R-Squared.

        Only given if the model has an intercept and family Normal.

        :return: R-squared for the regression.
        :rtype: float


        """
        if self.has_intercept and self.family == 'NORMAL':
            if (self.sse is not None) and (self.total_sum_of_squares is not None):
                return 1.0 - self.sse / self.total_sum_of_squares
            else:
                raise ModelError('glm: no sse or tss defined')
        else:
            return None
    
    @property
    def total_sum_of_squares(self):
        """
        Total sum of squares.

        Only returned for family Normal.

        :return: total sum of squares
        :rtype: float


        """
        return self.__tss
    
    @property
    def fitted(self):
        """
        Fitted values.

        :return: fitted values from the regression
        :rtype: numpy vector (*n* by 1)


        """
        return self.__fitted
    
    @property
    def sse(self):
        """
        Sum of squared error from regression fit.

        Only returned for family Normal.

        :return: sum of squared error
        :rtype: float


        """
        return self.__sse
    
    @property
    def mse(self):
        """
        Mean squared error from regression fit.

        Only returned for family Normal.

        :return: mean squared error
        :rtype: float


        """
        return self.__mse
    
    @property
    def t_table(self):
        """
        Table consisting of:

        - regression coefficients
        - standard errors
        - t-statistics
        - p-values
        - ridge parameters, if these have been specified

        :return: table
        :rtype: pandas DataFrame


        """
        return self.__ttable
    
    # no setter property
    @property
    def parameters(self):
        """
        Regression parameters

        :return: regression parameters
        :rtype: pandas Series


        """
        return self.__parameters
    
    def plot_effects(self, effect = None, wait=True):
        """
        Plots the marginal effect of a factor in the model.
        If *effect* is not specified, each factor is plotted.
        
        :param effect: A single effect to plot.  If omitted, all effects are plotted.
        :type effect: str
        :param wait: if True, then waits for a keypress after the plot
        :type wait: bool
        :return: plot
        :rtype: matplotlib plot


        """

        if (effect is not None):
            try:
                var_type = self.factors[effect]
            except:
                raise KeyError('glm: ' + effect + ' is not a factor in the model')
            self.__plot_single(effect,wait)
        else:
            for fx in list(self.factors.keys()):
                self.__plot_single(fx,wait)

    def __plot_single(self,effect,wait=True):
        """
        Plots the marginal effect of a factor in the model.
        
        :param effect: name of the effect (factor) in the model to plot.
        :type effect: str
        :param wait: if True, will wait for a keypress before continuing
        :type wait: bool
        :return: plot
        :rtype:  matplotlib plot
        
        
        """
        var_type = self.factors[effect]['type']
        if var_type == 'DIRECT':
            x_values = self.df[effect].quantile([0.01,.99])
            y_values = self.parameters[effect] * x_values
            plt.xlabel(effect + ' Value')
            plt.ylabel('Marginal Effect')
            plt.title('Marginal Effect of ' + effect)
            plt.plot(x_values,y_values)
            if wait:
                plt.waitforbuttonpress()
                plt.close()
        else:
            if var_type == 'SPLINES: HATS':
                x_values = self.factors[effect]['knots']
                z = self.parameters.index
                y_values_x = self.parameters[[(lambda x: x.find(effect) >= 0)(x) for x in z]]
                omit = self.factors[effect]['omit']
                if omit is None:
                    y_values = y_values_x
                else:
                    y_values = np.zeros(x_values.size)
                    y_values[np.arange(x_values.size)!=omit] = y_values_x
                plt.xlabel(effect + ' Value')
                plt.ylabel('Marginal Effect')
                plt.title('Marginal Effect of ' + effect)
                plt.plot(x_values,y_values)
                if wait:
                    plt.waitforbuttonpress()
                    plt.close()
            else:
                if var_type == 'SPLINES: LINEAR':
                    qs = self.df[effect].quantile([0.01, .99])
                    x_values = np.arange(qs[0.01],qs[0.99],(qs[0.99]-qs[0.01])/100)
                    knots = self.factors[effect]['knots']
                    z = self.parameters.index
                    y_values_x = self.parameters[[(lambda x: x.find(effect) >= 0)(x) for x in z]]
                    yh = np.zeros(x_values.size)
                    omit = self.factors[effect]['omit']
                    if (omit is None) or (omit == 0):
                        y_values = y_values_x
                        yh = y_values[0] * x_values
                    else:
                        y_values = np.zeros(knots.size)
                        y_values[np.arange(x_values.size) != omit] = y_values_x
                    for (j,knot) in enumerate(knots):
                        i = x_values >= knot
                        yh[i] += (x_values[i] - knot) * y_values[j+1]
                    plt.plot(x_values, yh)
                    plt.xlabel(effect + ' Value')
                    plt.ylabel('Marginal Effect')
                    plt.title('Marginal Effect of ' + effect)
                    if wait:
                      plt.waitforbuttonpress()
                      plt.close()
                else:
                    if var_type == 'CATEGORICAL':
                        z = self.parameters.index
                        labels = self.factors[effect]['levels']
                        x_values = np.arange(0,labels.size,1)
                        y_values_x = self.parameters[[(lambda x: x.find(effect) >= 0)(x) for x in z]]
                        omit = self.factors[effect]['omit']
                        if omit is None:
                            y_values = y_values_x
                        else:
                            y_values = np.zeros(x_values.size)
                            y_values[labels!=omit] = y_values_x
                        plt.xticks(x_values,labels,rotation=70)
                        plt.xlabel(effect + ' Value')
                        plt.ylabel('Marginal Effect')
                        plt.title('Marginal Effect of ' + effect)
                        plt.plot(x_values,y_values,'o')
                        if wait:
                            plt.waitforbuttonpress()
                            plt.close()

    def __glm_fitter(self):
        """Fit a linear regression or logistic regression model
        Populates these fields:
            self.__xpx
            self.__xpxI
            self.__parameters
            self.__fitted
            self.__mse
            self.__sse
            self.__tss
            self.__ttable
            self.__hat_diagonal
            self.__KS
            self.__f_statistic
            self.__chisq_statistic


        """
        x = self.x
        y = self.y
        if self.wts is not None:
            sqrtwts = np.sqrt(self.wts)
            y = np.multiply(y, sqrtwts)
            x = np.multiply(x, sqrtwts)
        self.__xpx = x.T * x
        if self.ridge is not None:
            self.__xpx += np.diag((np.squeeze(np.asarray(self.ridge))))
        # is the model over-specified?
        r = linalg.matrix_rank(self.__xpx)
        #if r < self.p:
        #    raise ModelError('glm: design matrix not full rank')
        # linear regression
        if self.family == 'NORMAL':
            xpy = x.T * y
            self.__xpxI = self.__xpx.I
            beta = self.__xpxI * xpy
            if self.__restriction_coefficients is not None:
                zpz_i = (self.__restriction_coefficients * self.__xpxI * self.__restriction_coefficients.T).I
                beta = beta - self.__xpxI * self.__restriction_coefficients.T * zpz_i * (
                    self.__restriction_coefficients * beta - self.__restriction_values)
            self.__fitted = x * beta
            res = y - x * beta
            sigma2 = float(res.T * res)
            self.__sse = sigma2
            self.__tss = float((y - y.mean()).T * (y - y.mean()))
            self.__mse = sigma2 / float(self.model_degrees_freedom)
            beta = pd.DataFrame(beta)
            beta.index = self.x_column_names
            beta.columns = ['beta']
            if self.__restriction_coefficients is not None:
                vcv = self.__xpxI - self.__xpxI * self.__restriction_coefficients.T * zpz_i * self.__restriction_coefficients * self.__xpxI
                se = np.sqrt(self.__mse * np.diag(vcv))
            else:
                se = np.sqrt(self.__mse * np.diag(self.__xpxI))
            not0 = (se > 1e-7)
            t = np.zeros(beta.shape[0])
            t[not0] = beta.beta[not0] / se[not0]
            beta['se'] = se
            beta['t'] = t
            beta['pvalue'] = 2.0 * (1 - stats.t.cdf(abs(t), self.model_degrees_freedom))
            if not np.all(not0):
                beta.t[np.logical_not(not0)] = np.NaN
                beta.pvalue[np.logical_not(not0)] = np.NaN
            self.__ttable = beta
            self.__parameters = self.t_table['beta']
            if self.has_intercept and (self.p > 1):
                f = (self.total_sum_of_squares - self.sse) / (self.p - 1)
                f /= self.mse
                self.__f_statistic = f
            if self.n <= 1000:
                self.__hat_diagonal = np.diag(x * self.__xpxI * x.T)
        # logistic regression via IRWLS
        if self.family == 'BINOMIAL':
            from modeling_tools.functions import ks_calculate
            wt = np.empty(self.n)
            wt.fill(1.)
            y1 = y.copy()
            for j in range(50):
                # Form x' * w * x and x' * w * y
                sqrt_wt = np.sqrt(wt.reshape(self.n, 1))
                xx = np.multiply(self.x, sqrt_wt)
                xy = np.multiply(y1, sqrt_wt)
                xpy = xx.T * xy
                self.__xpx = xx.T * xx
                if self.ridge is not None:
                    self.__xpx += np.diag((np.squeeze(np.asarray(self.ridge))))
                self.__xpxI = self.__xpx.I
                b = self.__xpxI * xpy
                if self.__restriction_coefficients is not None:
                    zpz_i = (self.__restriction_coefficients * self.__xpxI * self.__restriction_coefficients.T).I
                    b -= self.__xpxI * self.__restriction_coefficients.T * zpz_i * (
                        self.__restriction_coefficients * b - self.__restriction_values)
                yhat = x * b
                # Sometimes, the algorithm will run away causing yhat to grow to the point exp(yhat) overflows.
                # This means, of course, that some of the parameters are growing large.
                # So, if we are on this track, pull it back in by specifying a ridge which has the effect of
                # shrinking the parameters toward 0.
                aymax = yhat.max()
                if aymax > 12:
                    ridge = np.matrix(np.zeros(self.x.shape[1])).T
                    ridge.fill(0.1)
                    if self.ridge is None:
                        self.__ridge = ridge
                    else:
                        self.__ridge += ridge
                if j > 0:
                    delta = (abs(b-bold)).max()
                    if (j==49) or (delta < 0.0001):
                        break
                yhat_exp = np.exp(yhat)
                p0 = yhat_exp / (1 + yhat_exp)
                wt = np.squeeze(np.asarray(p0 / (1 + yhat_exp)))
                y1 = np.reshape(np.squeeze(np.asarray(y)) / wt, (self.n, 1)) - 1 - yhat_exp + yhat
                bold = b
                if j == 50:
                    raise ModelError('glm: outer iterations exceeded')
            self.__fitted = x * b
            self.__fitted = np.exp(self.__fitted)
            self.__fitted /= 1.0 + self.__fitted
            beta = pd.DataFrame(b)
            beta.index = self.x_column_names
            beta.columns = ['beta']
            self.__sse = self.n - self.p
            # scale parameter taken to be 1 since weights derived to actually be the variance
            self.__mse = 1
            self.__tss = None
            if self.__restriction_coefficients is not None:
                vcv = self.__xpxI - self.__xpxI * self.__restriction_coefficients.T * zpz_i * self.__restriction_coefficients * self.__xpxI
                se = np.sqrt(np.diag(vcv))
            else:
                se = np.sqrt(np.diag(self.__xpxI))
            not0 = (se > 1e-7)
            t = np.zeros(beta.shape[0])
            t[not0] = beta.beta[not0] / se[not0]
            #se = np.sqrt(np.diag(self.__xpxI))
            #t = beta.beta / se
            # likelihood ratio statistic
            y = np.squeeze(np.asarray(self.y))
            p0 = np.squeeze(np.asarray(p0))
            ll = np.where(y == 1, np.log(p0), np.log(1-p0))
            l1 = ll.sum()
            p = y.mean()
            l2 = (y.sum()) * math.log(p) + (self.n - y.sum()) * math.log(1-p)
            chisq = -2 * (l2 - l1)
            self.__chisq_statistic = chisq
            beta['se'] = se
            beta['t'] = t
            beta['pvalue'] = 2.0 * (1 - stats.t.cdf(abs(t), self.model_degrees_freedom))
            if not np.all(not0):
                beta.t[np.logical_not(not0)] = np.NaN
                beta.pvalue[np.logical_not(not0)] = np.NaN
            self.__ttable = beta
            self.__parameters = self.t_table['beta']
            wt = np.matrix(wt).T
            x = np.multiply(x, np.sqrt(wt))
            if self.n <= 1000:
                self.__hat_diagonal = np.diag(x * self.__xpxI * x.T)
            self.__KS = ks_calculate(self.fitted, self.y)
    
    def predict(self, dfnew):
        """
        Generates the predicted values of the fit model on a new DataFrame.

        The DataFrame of new data, *dfnew*, must have the columns required to run the model (*i.e.* the independent variables).

        :param dfnew: new data to run the model on.
        :type dfnew: pandas DataFrame
        :return: model output on the new data.
        :rtype: numpy column vector.


        """
        if self.__parameters is None:
            raise ModelError("glm: can't use predict before the model is built")
        d = DataClass(dfnew)
        formula = self.formula.split('~')[1]  # don't need a dependent variable
        d.formula = formula
        d.family = self.family
        d.event = self.event
        d.build_design()
        p = d.x * np.matrix(self.parameters).T
        if self.family == 'BINOMIAL':
            p = np.exp(p)
            p /= 1.0 + p
        return p
    
    @staticmethod
    def __get_term(eqn):
        """
        This function is used in parsing restrictions.
        It pulls a single term off from the equation, returns that term and the remaining equation
        
        :param eqn: restriction equation (may be partial)
        :type eqn: str
        :return: first term in eqn and the remaining eqn with term removed
        :rtype: list of str
        
        
        """
        plus = eqn[1:len(eqn)].find('+')
        if plus >= 0:
            plus += 1
        minus = eqn[1:len(eqn)].find('-')
        if minus >= 0:
            minus += 1
        if plus >= 0 and minus >= 0:
            to_take = min(plus, minus)
        else:
            if plus >= 0 or minus >= 0:
                to_take = max(plus, minus)
            else:
                to_take = len(eqn)
        term = eqn[0:to_take]
        eqn = eqn[to_take:len(eqn)]
        if len(eqn) > 0:
            if eqn[0] == '+':
                eqn = eqn[1:len(eqn)]
        return term, eqn
    
    @staticmethod
    def __split_term(term):
        """
        This function is used in parsing restrictions.
        It takes a single term and splits off the coefficient from the variable name
        
        :param term: single term of the restrcionts
        :type term: str
        :return: restriction coefficient and variable name
        :rtype: list of str
        
        
        """
        star = term.find('*')
        # no coefficient....assume it is -1 or 1.
        if star < 0:
            if term[0] == '-':
                return [-1, term.replace(' ', '').replace('-', '')]
            return [1, term.replace(' ', '')]
        if star == 0:
            raise ModelError('glm: restriction has a * but no coefficient')
        try:
            coef = float(term[0:star])
        except:
            raise ModelError('glm: restriction coefficents must be constants')
        variable_name = term[(star + 1):len(term)].replace(' ', '')
        return [coef, variable_name]
    
    def __restriction_to_row_vector(self, xcols, restriction):
        """
        Restrictions: takes a single restriction and returns a row vector of coefficients against x and the value
        of the restriction.
        
        :param xcols: The names of the columns of the design matrix
        :type xcols: list of str
        :param restriction: single restriction
        :type restriction: str
        :return: array of restriction coefficients that has the same # of columns as x, RHS value
        :rtype: list
        
        
        """
        eq = restriction.find('=')
        if eq < 0:
            raise ModelError('glm: restrictions must be the form of an equality (no = found)')
        left_hand = restriction[0:eq]
        right_hand = restriction[(eq + 1):len(restriction)]
        try:
            equal_to_value = float(right_hand)
            eqn = left_hand
        except ValueError:
            try:
                equal_to_value = float(left_hand)
                eqn = right_hand
            except ValueError:
                raise ModelError('glm: one side of the restriction must be a constant')
        z = np.matrix(np.zeros(xcols.shape[0]))
        while len(eqn) > 0:
            [term, eqn] = self.__get_term(eqn)
            [coefficient, variable_name] = self.__split_term(term)
            location = xcols == variable_name
            if location.sum() != 1:
                raise ModelError('glm: variable: ' + variable_name + ' not in the model')
            z = np.where(location, coefficient, z)
        return z, np.matrix(equal_to_value)
    
    def __build_restriction_matrices(self, restriction_list):
        """
        Runs through the list of restrictions and creates a matrix of coefficients against x and
        a column vector of the RHS (what the linear combination must equal).
        
        :param restriction_list: list of restrictions on the regression coefficients
        :type restriction_list:  list of str
        :return:  self.__restriction_coefficients, self.__restriction_values
        :rtype: numpy matrix, numpy vector
        
        
        """
        restriction_coefficients = None
        restriction_values = None
        for restriction in restriction_list:
            z, c = self.__restriction_to_row_vector(self.x_column_names, restriction)
            if restriction_coefficients is None:
                restriction_coefficients = z
                restriction_values = c
            else:
                restriction_coefficients = np.append(restriction_coefficients, z, axis=0)
                restriction_values = np.append(restriction_values, c, axis=0)
        r = linalg.matrix_rank(restriction_coefficients)
        if r < restriction_coefficients.shape[0]:
            raise ModelError('glm: restrictions are linearly dependent')
        self.__restriction_coefficients = restriction_coefficients
        self.__restriction_values = restriction_values
    
    def implement(self, file_name, function_name='model', indent_level=0):
        """
        :param file_name: file to create for the implementation
        :type file_name: str
        :param function_name: name to call the function being implemented
        :type function_name: str
        :param indent_level: how much to indent the code at the 'def' level
        :type indent_level: int
        :return: None
        
        
        """
        try:
            output_file = open(file_name, 'w')
        except FileNotFoundError:
            raise FileNotFoundError('glm: cannot open the file: ' + file_name)
        except FileExistsError:
            raise FileExistsError('glm: cannot open file file for writing: ' + file_name)
        indent = ''
        for j in range(indent_level): indent += ' '
        output_file.write('from modeling_tools.functions import categorical_to_design, linear_splines_basis1, linear_splines_basis2\n')
        output_file.write('import numpy as np\n')
        output_file.write(indent + 'def ' + function_name + '(df_in):\n')
        indent += '    '
        for (line, factor) in self.implementation_skeleton:
            if factor is None:
                output_file.write(indent + line + '\n')
            else:
                index = self.x_column_names == factor
                if index.sum() != 1:
                    raise ValueError('glm: parameter not found')
                param = self.parameters[index]
                line1 = indent + line + str(float(param))
                output_file.write(line1 + '\n')
        if self.family == 'BINOMIAL':
            output_file.write(indent + 'fn = np.exp(fn)\n')
            output_file.write(indent + 'fn /= (1+fn)\n')
        output_file.write(indent + 'return fn\n')
