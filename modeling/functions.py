import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.pyplot as plt



def smooth_linear_splines(x, y, xtest, num_turn, holdout_rate = 0.5, wts=None):
    """
    
    This function uses linear splines to smooth a scatterplot. The input x is a matrix of linear splines using
    basis2.  The first two columns are the constant and linear terms.  The remaining terms, that provide the
    nonlinearity, are shrunk toward zero using ridge regression.
    
    The ridge parameter is selected to minimize the sum of squared error on a holdout sample.
    
    See https://www.whitman.edu/Documents/Academics/Mathematics/Griggs.pdf section 6.1
    
    :param x: matrix of linear (basis2) splines
    :type x: numpy matrix
    :param y: dependent variable
    :type y: numpy array or numpy column matrix or pandas series
    :param xtest: splines at an array of points at which to evaluate the function to assess smoothness
    :type xtest: numpy matrix
    :param num_turn: smoothness criterion. Number of turns permitted in function.
    :type num_turn: float
    :param holdout_rate: fraction of observations to hold out for validation to select shrinkage
    :type holdout_rate: float
    :param wts: weights for the fit
    :type wts: numpy array
    :return: yhat: fitted values, beta: paramaeters, ridge: ridge parameter
    :rtype: dict
    
    """

    from modeling.data_class import DataError
    import math
    x = x.copy()
    y = y.copy()
    if (str(type(y)).find('numpy.ndarray') >= 0) or (str(type(y)).find('pandas.core.series') >= 0):
        y = np.matrix(y).T
    else:
        if str(type(y)).find('numpy.matrixlib.defmatrix.matrix') < 0:
            raise DataError('smooth_hats: y must be numpy array or column vector')
    if str(type(x)).find('pandas.core.frame.DataFrame') >= 0:
        x = np.matrix(x)
    if str(type(x)).find('numpy.matrixlib.defmatrix.matrix') == 0:
        raise DataError('smooth_hats: x must be numpy matrix')
    if x.shape[0] != y.shape[0]:
        raise DataError('smoth_hats: x and y must have the same number of rows')
    #randomly divide into two pieces: fit (to estimate beta) and val (validate--used to choose the ridge parameter)
    u = np.random.uniform(0,1,x.shape[0])

    ifit = u >= holdout_rate
    ival = np.logical_not(ifit)
    if not (wts is None):
        sqrtwts = np.sqrt(wts)
        sqrtwts = np.matrix(sqrtwts).T
        y1 = np.multiply(y, sqrtwts)
        x1 = np.multiply(x, sqrtwts)
        sqrtwtsfit = sqrtwts[ifit]
        sqrtwtsval = sqrtwts[ival]
    else:
        y1 = y
        x1 = x

    xfit = x1[ifit]
    xlinear_fit = x1[ifit][:,[0,1]]
    xpx_linear = xlinear_fit.T * xlinear_fit
    xpx = xfit.T * xfit
    #if linalg.matrix_rank(xpx) < xpx.shape[0]:
    #    raise DataError('smooth_hats: x is not full column rank')
    yfit = y1[ifit]
    xpy_linear = xlinear_fit.T * yfit
    xpy = xfit.T * yfit
    xlinear_val = x[ival][:,[0,1]]
    xval = x[ival]
    yval = y[ival]
    # create the ridge matrix.  There is no ridge parameter for the intercept & linear component
    tmp = np.empty(x.shape[1])
    tmp.fill(1)
    tmp[0] = 0
    tmp[1] = 0
    ridge_matrix = np.diag(tmp)
    #get a sense of the scale of the diag values
    maxdiag = np.squeeze(np.array(xpx.diagonal()))[2:xpx.shape[0]].max()
    # the purpose of this section is to find values that bracket the minimum...
    # So, we're either going to start with ridge=0 and then increase until the holdout SSE goes up or
    # we'll start the a big ridge (so nearly the linear fit) and work backward.
    beta = xpx_linear.I * xpy_linear
    res = yval - xlinear_val * beta
    if not (wts is None):
        res = np.multiply(res, sqrtwtsval)
    sse_linear = float(res.T * res)
    res = y[ifit] - x[ifit][:,[0,1]] * beta
    if not (wts is None):
        res = np.multiply(res, sqrtwtsfit)
    sse_linearF = float(res.T * res)

    beta = xpx.I * xpy
    res = yval - xval * beta
    if not (wts is None):
        res = np.multiply(res, sqrtwtsval)
    sse_wiggly = float(res.T * res)
    res = y[ifit] - x[ifit] * beta
    if not (wts is None):
        res = np.multiply(res, sqrtwtsfit)
    sse_wigglyF = float(res.T * res)
    num_df = x.shape[1] - 2
    den_df = x.shape[0] - x.shape[1]
    f = ((sse_linearF - sse_wigglyF)/num_df)/(sse_wigglyF/den_df)
    pvalue = 1 - stats.f.cdf(f, num_df, den_df)

    xs = np.zeros(3)
    ys = np.zeros(3)
    step = min(maxdiag/1000,1)  # this is the value to step the ridge parameter
    #I've observed that the solutions tend to be at one extreme or the other: either very smooth
    #or very wiggly.  Require a lot of evidence for a large ridge:
    if (sse_linear < sse_wiggly) and (pvalue > 0.01):
        # start from linear answer
        xs[0] = 1000*maxdiag
        ys[0] = sse_linear
        direction = -1
    else:
        # start from very wiggly answer (ridge = 0)
        #num_turn = max(5,0.15*x.shape[1])
        xs[0] = 0
        c = 0.1 * math.log(1 / (maxdiag))
        cnt = 10
        i1 = range(0, xtest.shape[0] - 2)
        i2 = range(1, xtest.shape[0] - 1)
        i3 = range(2, xtest.shape[0])
        while True:
            step = maxdiag * math.exp(c * cnt)
            beta = (xpx + xs[0] * ridge_matrix).I * xpy
            ytest = xtest * beta
            delta1 = np.logical_and(ytest[i3] - ytest[i2]<0,ytest[i2]-ytest[i1]>0)
            delta2 = np.logical_and(ytest[i3] - ytest[i2]>0,ytest[i2]-ytest[i1]<0)
            chg = np.logical_or(delta1,delta2).sum()
            #plt.plot(xtest[:,1],ytest)
            #plt.waitforbuttonpress()
            #plt.close()
            if (chg <= num_turn) or (cnt == 0):
                break
            cnt -= 1
            xs[0] += step
        
        ys[0] = sse_wiggly
        direction = 1
    #plt.plot(xtest[:,1],ytest)
    #plt.waitforbuttonpress()
    #plt.close()
    xs[1] = xs[0]
    #The parameter c controls the exponential grid search
    c = 0.1 * math.log(1 / (1000 * maxdiag))
    cte = True
    cnt = 10
    # The idea is to find 3 points for the ridge parameter: a low point with higher sse values on either side.
    # The first point is the edge.  The need is not to be precise but to set up the quadratic optimization for
    # success.
    #This is doing a grid search.  Experimentation showed that often there is a decline near the edge, so start
    #looking near the edge. But this may not be true...so need to be able to cover a large span.  So, look at a set
    #of grids that are exponentially determined.
    while True:
        step = 1000 * maxdiag * direction * math.exp(c * cnt)
        xs[1] = xs[0] + step
        beta = (xpx + xs[1] * ridge_matrix).I * xpy
        res = yval - xval * beta
        if not (wts is None):
            res = np.multiply(res, sqrtwtsval)
        sse = float(res.T * res)
        if sse < ys[0]:
            ys[1] = sse
            break
        cnt -= 1
        if cnt == -1:
            cte = False
            #pick edge solution--never found a lower SSE
            xs[1] = xs[0]
            break
    #now find a point that has SSE larger than ys[1]...this should exist
    if cte:
        cnt = 10
        xs[2] = xs[1]
        if direction == 1:
            width = 1000*maxdiag - xs[1]
        else:
            width = xs[1]
        c = 0.1 * math.log(1 / width)
        while (ys[2] < ys[1]):
            step = width*direction*math.exp(c*cnt)
            xs[2] = xs[1] + step
            beta = (xpx + xs[2] * ridge_matrix).I * xpy
            res = yval - xval * beta
            if not (wts is None):
                res = np.multiply(res, sqrtwtsval)
            ys[2] = float(res.T * res)
            cnt -= 1
            if cnt < 0:
                # could not find one...so the minimum at xs[1] will be used.
                cte = False
                break
    # Quadratic optimization
    if cte:
        #This routine requires xs[0] < xs[1] < xs[2].
        if direction == -1:
            tmp = xs[0]
            xs[0] = xs[2]
            xs[2] = tmp
            tmp = ys[0]
            ys[0] = ys[2]
            ys[2] = tmp
            
        ok = True
        cnt = 0
        while ok:
            cnt += 1
            # find the quadratic that goes through the 3 points (xs,ys)
            f = (xs[0] - xs[2]) / (xs[0] - xs[1])
            a = ((ys[0] - ys[1]) * f - (ys[0] - ys[2])) / ((xs[0] ** 2 - xs[1] ** 2) * f - (xs[0] ** 2 - xs[2] ** 2))
            # if a <=0 then the quadratic has a max not a min.  In this case, bisect the biggest interval of:
            # xs[0] to xs[1] and xs[1] to xs[2]
            if a <= 0.0:
                if (xs[2] - xs[1]) > (xs[1] - xs[0]):
                    xnew = (xs[1] + xs[2]) / 2
                else:
                    xnew = (xs[0] + xs[1]) / 2
            else:
                b = (ys[0] - ys[2] - a * (xs[0] ** 2 - xs[2] ** 2)) / (xs[0] - xs[2])
                xnew = -b/(2*a)
            #if overstep our bounds, then bisect the interval in the direction the quadratic indicates
            if xnew < min(xs):
                xnew = (xs[0] + xs[1])/2
            if xnew > max(xs):
                xnew = (xs[1] + xs[2]) / 2
            #find sse of our new point
            beta = (xpx + xnew * ridge_matrix).I * xpy
            res = yval - xval * beta
            if not (wts is None):
                res = np.multiply(res, sqrtwtsval)
            sse = float(res.T * res)
            #We're going to stop if our x values aren't changing much relative to the width of the interval we started
            #or the range of the sse triplets is small (not much to gain)
            ok = (abs(xnew - xs[1]) > 0.01 ) and ((ys[2] - ys[1]) > 0.0001*ys[1]) \
                 and ((ys[0] - ys[1]) > 0.0001*ys[1])
            #select a new triplet incorporating (xnew,sse). To see what  this is doing check out the graph here:
            #https://en.wikipedia.org/wiki/Golden-section_search
            if xnew > xs[1]:
                if sse > ys[1]:
                    xs[2] = xnew
                    ys[2] = sse
                else:
                    xs[0] = xs[1]
                    ys[0] = ys[1]
                    xs[1] = xnew
                    ys[1] = sse
            else:
                if sse > ys[1]:
                    xs[0] = xnew
                    ys[0] = sse
                else:
                    xs[2] = xs[1]
                    ys[2] = ys[1]
                    xs[1] = xnew
                    ys[1] = sse
            
            # stop if 5 iterations
            if cnt == 5:
                ok = False
    #The final ridge parameter is xs[1]
    beta = (xpx + xs[1] * ridge_matrix).I * xpy

    yh = x[ifit] * beta
    xf = x[ifit][:,1]
    #plt.plot(xf, yfit, 'ro')
    #plt.plot(xf,yh,'bo')
    xa = np.arange(0, 1, .01)
    xb = np.zeros(xa.shape[0])
    ya = 10*(xa-0.5)**2
    ya = ya - ya.mean() + yh.mean()
    #plt.plot(xa, ya,'black')
    #plt.waitforbuttonpress()
    #plt.close()

    yhat = x * beta
    to_return = {'yhat': yhat, 'beta': beta, 'ridge': xs[1]}
    return to_return


def linear_splines_basis1(x, knots, omit=None):
    """
    Create basis functions based on linear splines.
    If there are k knots, then there are k basis functions.  The basis functions are number 0 to (k-1).
    These basis functions are sometimes called 'hats'.  There is no intercept or overall linear term.
    The output spline will connect the dots of the parameters applied to each basis function.

    The form of the basis functions is:
    
         b0(x) = 1 if x <= k[0],
                 (k[1] - x)/(k[1] - k[0]) if k[0] <= x <= k[1]
                 0 if x >= k[1]
                 
         bj(x) = 0 if x <= k[j-1]
                 (x - k[j-1]) / (k[j] - k[j-1]), if k[j-1] <= x <= k[j]
                 (k[j+1] - x) / (k[j+1] - k[j]), if k[j] <= x <= k[j+1]
                 0 if x >= k[j+1]
                 for 0 < j < n
                 
         bn(x) = 0 if x <= k[n-1]
                 (x - k[n-1]) / (k[n] - k[n-1]), if k[n-1] <= x <= k[n]
                 1 if x >= k[n]
                 
         for knot points k[0] < k[1] < ... < k[n]
         
    An important difference between basis1 and basis2 is that basis1 is flat after k[n] and before k[0].
    The basis2 result is basis2 is linear.

    :param x: variable from which to calculate the values of the linear spline basis functions.
    :type x: pandas series: type float
    :param knots: knot points of the splines
    :type knots: numpy array
    :param omit: basis function to omit from the return (0=first basis function). Returns all if None.
    :type omit: int
    :return: basis function values.
    :rtype: pandas dataframe. columns are named <x name> <basis function number>


    """
    from modeling.data_class import DataError
    if str(type(knots)).find('list') > 0:
        knots = np.array(knots)
    knots.sort()
    if not (omit is None):
        if omit >= knots.size or omit < 0:
            raise DataError('linear_splines_basis1: at to drop out of range:' + str(omit))
    # Set up column names
    nms = []
    for n in range(0, knots.size):
        nms += [x.name + str(n)]
    
    df_out = pd.DataFrame(index=range(0, x.size), columns=nms, dtype='float64')  # Setup output DataFrame
    df_out.fillna(0, inplace=True)
    
    i = x < knots[0]
    df_out[x.name + '0'][i] = 1
    
    n = knots.size - 1
    i = x >= knots[n]
    df_out[x.name + str(n)][i] = 1
    for i in range(1, knots.size):
        ibool = (x >= knots[i - 1]) & (x < knots[i])
        wts = (x[ibool] - knots[i - 1]) / (knots[i] - knots[i - 1])
        df_out.loc[ibool, x.name + str(i - 1)] = 1 - wts
        df_out.loc[ibool, x.name + str(i)] = wts
    
    # drop column specified
    if not (omit is None):
        target = x.name + str(omit)
        keep_cols = df_out.columns[df_out.columns != target]
        df_out = df_out[keep_cols]
    return df_out


def linear_splines_basis2(x, knots, omit=None):
    """
    Create basis functions based on linear splines.
    If there are n knots, there are k+2 basis functions.
    
    The basis functions are:
    
        - b0(x) = 1
        - b1(x) = x
        - bn(x)
        
            - = 0 if x < k[j]
            - = x - k[j], if x > k[j]
    
    An important difference between basis1 and basis2 is that basis1 is flat after k[n] and before k[0].
    The basis2 result is basis2 is linear.

    :param x: variable from which to calculate the values of the linear spline basis functions.
    :type x: pandas series: type float
    :param knots: knot points of the splines
    :type knots: numpy array
    :param omit: basis function to omit from the return (0=first basis function). Returns all if None.
    :type omit: int
    :return: basis function values.
    :rtype: pandas dataframe. columns are named <x name> <basis function number>


    """
    from modeling.data_class import DataError
    if str(type(knots)).find('list') > 0:
        knots = np.array(knots)
    knots.sort()
    if not (omit is None):
        if omit >= knots.size or omit < 0:
            raise DataError('linear_splines_basis2: basis to drop out of range:' + str(omit))
    # Set up column names
    nms = []
    for n in range(0, knots.size+2):
        nms += [x.name + str(n)]
    
    df_out = pd.DataFrame(index=range(0, x.size), columns=nms, dtype='float64')  # Setup output DataFrame
    df_out.fillna(0, inplace=True)

    #fill in first two columns, these follow a different pattern
    ones = np.empty(x.shape[0])
    ones.fill(1)
    df_out[nms[0]] = ones
    df_out[nms[1]] = x
    
    for i in range(knots.size):
        ibool = (x >= knots[i])
        wts = x[ibool] - knots[i]
#        df_out[x.name + str(i + 2)][ibool] = wts
        df_out.loc[ibool, x.name + str(i + 2)] = wts

    # drop column specified
    if not (omit is None):
        target = x.name + str(omit)
        keep_cols = df_out.columns[df_out.columns != target]
        df_out = df_out[keep_cols]
    return df_out


def categorical_to_design(cat_var, omit=None, error_out=True):
    """
    Create the design matrix of indicator variables from an input categorical variable.
    if omit == None, then each distinct value of cat_var will receive a column in the return.

    :param cat_var: categorical variable that forms the basis of design matrix
    :type cat_var: pandas series: can be of any type.
    :param omit: optional value of cat_var to omit from the design matrix (to avoid collinearity)
    :type omit: object: same type as cat_var
    :param error_out: if True, throws an error if there is a problem
    :type error_out: bool
    :return: design matrix
    :rtype: pandas data frame: column names have the format 'cat_var' + : + <value of cat_var indicator indicates>


    """
    from modeling.data_class import DataError

    if str(cat_var.dtype) != 'category':
        cat_var = cat_var.astype('str')
    vals = cat_var.value_counts().index.astype('str').sort_values()
    if vals.size == 1 and error_out:
        raise DataError('categorical_to_design: series ' + cat_var.name + ' has only 1 level')
    target_vals = vals
    if not (omit is None):
        omit = str(omit)
        i = vals == omit
        if i.sum() == 0 and error_out:
            raise DataError('categorical_to_design: category does not have level: '
                            + omit + ' or conflicting data types')
        target_vals = vals[vals != omit]
    #
    # Set up output DataFrame
    df_out = pd.DataFrame(np.zeros((cat_var.size, target_vals.size)))
    nms = []
    for n in target_vals:
        nms += [cat_var.name + ':' + str(n)]
    df_out.columns = nms
    for n in target_vals:
        i = cat_var == n
        df_out.ix[i, cat_var.name + ':' + str(n)] = 1
    #
    ret = {'df_out': df_out, 'levels': np.asarray(vals)}
    return ret


def ks_calculate(score_variable, binary_variable, plot=False, wait=True, xlab='Score', ylab='CDF', title='KS Plot', subtitle=None):
    """
    Calculates the KS (Kolmogorov Smirnov) distance between two cdfs.  The KS statistic is 100 times the
    maximum vertical difference between the two cdfs

    The single input score_variable contains values from the two populations.  The two populations are distinguished
    by the value of binvar (0 means population A, 1 means population B).
    
    Optionally, the plot of the CDF of score variable for the two values of binary_variable may be plotted.

    :param score_variable: continuous variable from the logistic regression
    :type score_variable: pandas series, numpy array or numpy vector
    :param binary_variable: binary outcome (dependent) variable from the logistic regression
    :type binary_variable: numpy array or numpy vector
    :param xlab: label for the x-axis (score variable), optional
    :type xlab: str
    :param ylab: label for the y-axis (binary variable), optional
    :type ylab: str
    :param title: title for the plot, optional
    :type title: str
    :param subtitle: subtitle for the plot, optional (default=None)
    :type subtitle: str
    :param wait: if True waits for a keypress after creating the plot, optional (default=True)
    :type wait: bool
    :return: KS statistic (0 to 100),
    :rtype: float


    """
    
    if str(type(score_variable)).find('numpy.ndarray') >= 0:
        score_variable = pd.Series(score_variable)
    else:
        if str(type(score_variable)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
            score_variable = pd.Series(np.squeeze(np.asarray(score_variable)))

    if str(type(binary_variable)).find('numpy.ndarray') >= 0:
        binary_variable = pd.Series(binary_variable)
    else:
        if str(type(binary_variable)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
            binary_variable = pd.Series(np.squeeze(np.asarray(binary_variable)))

    # divide the score_variable array by whether the binary variable is 0 or 1
    index0 = binary_variable == 0
    index1 = binary_variable == 1
    
    # sort the scores
    score0 = score_variable[index0]
    score0 = score0.sort_values()
    score0.index = np.arange(0, score0.shape[0])
    u0 = (np.arange(0,score0.shape[0])+1-0.5)/score0.shape[0]

    score1 = score_variable[index1]
    score1 = score1.sort_values()
    score1.index = np.arange(0, score1.shape[0])
    u1 = (np.arange(0,score1.shape[0])+1-0.5)/score1.shape[0]

    # interpolate these at common values
    delta = (score_variable.max() - score_variable.min())/100
    sc = np.arange(score_variable.min(), score_variable.max(), delta)
    
    ind0 = score0.searchsorted(sc)
    # it's possible that ind0 may have a value = score0.shape[0] (e.g. sc bigger than biggest score0)
    ind0[ind0 >= score0.shape[0]] = score0.shape[0] - 1
    uu0 = u0[ind0]

    ind1 = score1.searchsorted(sc)
    ind1[ind1 >= score1.shape[0]] = score1.shape[0] - 1
    uu1 = u1[ind1]
    
    ks = round(float(100.0 * max(abs(uu1 - uu0))), 1)

    if plot:
#        plt.plot(p0,u,'black',p1,u,'black')
        plt.plot(score0,u0,'black',score1,u1,'black')
        plt.plot(sc,uu0,'red',sc,uu1,'red')
        maxx = score_variable.max()
        plt.axis([0, maxx, 0, 1])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
    
        if subtitle is None:
            sub_title = "KS: " + str(ks)
        else:
            sub_title = subtitle + "\nKS: " + str(ks)
    
        plt.figtext( (maxx) / 2.0, 0.15, sub_title, ha='center')
        plt.show()
    
        if wait:
            plt.waitforbuttonpress()
            plt.close()

    return ks


def decile_plot(score_variable, binary_variable, xlab='Score', ylab='Actual', title='Decile Plot',
                plot_maximum=None, plot_minimum=None, confidence_level=0.95, correlation=0, subtitle=None, wait=True):
    """
    This function creates the so-called decile plot.  The input data (score_variable, binary_variable) is
    divided into 10 equal groups based on the deciles of score_variable.  Within each decile, the values of the
    two are averaged.  These 10 pairs are plotted.  A reference line is ploted.  Within each group a confidence
    interval is plotted as a vertical line.  The user may specify the confidence level and also the pair-wise
    correltion between the points (binary variable) within a decile.
    
    
    
    :param score_variable: continuous variable from the logistic regression
    :type score_variable: pandas series, numpy array or numpy column vector
    :param binary_variable: binary outcome (dependent) variable from the logistic regression
    :type binary_variable: pandas series, numpy array or numpy column vector
    :param xlab: label for the x-axis (score variable), optional
    :type xlab: str
    :param ylab: label for the y-axis (binary variable), optional
    :type ylab: str
    :param title: title for the plot, optional
    :type title: str
    :param plot_maximum: maximum value for the plot, optional
    :type plot_maximum: float
    :param plot_minimum: minimum value for the plot, optional
    :type plot_minimum: float
    :param confidence_level: confidence level for confidence intervals around each decile, optional (default = 0.95)
    :type confidence_level: float
    :param correlation: pair-wise correlation between data within each decile, optional (default=0)
    :type correlation: float
    :param subtitle: subtitle for the plot, optional (default=None)
    :type subtitle: str
    :param wait: if True waits for a keypress after creating the plot, optional (default=True)
    :type wait: bool
    :return: plot
    :rtype: N/A
    """

    if str(type(score_variable)).find('numpy.ndarray') >= 0:
        score_variable = pd.Series(score_variable)
    else:
        if str(type(score_variable)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
            score_variable = pd.Series(np.squeeze(np.asarray(score_variable)))

    if str(type(binary_variable)).find('numpy.ndarray') >= 0:
        binary_variable = pd.Series(binary_variable)
    else:
        if str(type(binary_variable)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
            binary_variable = pd.Series(np.squeeze(np.asarray(binary_variable)))

    bins = pd.qcut(score_variable, 10, labels=False, duplicates='drop')

    mscore = score_variable.groupby(bins).mean()
    counts = score_variable.groupby(bins).count()
    mbinary = binary_variable.groupby(bins).mean()
    vbinary = binary_variable.groupby(bins).var()

    # is the response binary?
    vals = np.unique(binary_variable)
    binary = (len(vals) == 2) and min(vals) == 0 and max(vals) == 1

    # Critical N(0,1) value for confidence intervals
    zcrit = stats.norm.isf((1.0 - confidence_level) / 2.0)

    if plot_maximum is None:
        # Want the x & y axes to have same range
        max_limit = max(max(mscore), max(mbinary))
        if binary:
            max_limit = round(max_limit + 0.05, 1)
        else:
            max_limit = round(max_limit * 1.05, 1)
    else:
        max_limit = plot_maximum  # User-supplied value
    
    if plot_minimum is None:
        # Want the x & y axes to have same range
        min_limit = min(mscore[0], mbinary[0])
        if binary:
            min_limit = round(min_limit - 0.05, 1)
        else:
            min_limit = round(min_limit * 0.95, 1)
    else:
        min_limit = plot_minimum  # User-supplied value

    # Reference line
    rxy = [min_limit, max_limit]
    # plot--deciles
    plt.plot(rxy, rxy, mscore, mbinary, 'ro')

    # Do confidence intervals
    for k in range(0, mbinary.shape[0]):
        if binary:
            variance = mbinary[k] * (1.0 - mbinary[k])
        else:
            variance = vbinary[k]
        width = zcrit * math.sqrt(variance * (1 + (1 + counts[k]) * correlation) / counts[k])
        plt.plot([mscore[k], mscore[k]], [mbinary[k] - width, mbinary[k] + width], 'r')
    
    # Make pretty
    plt.axis([min_limit, max_limit, min_limit, max_limit])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    
    # Add # of obs
    sub_title = np.str(score_variable.shape[0]) + ' Obs'
    sub_title += '\nconfidence: ' + str(confidence_level)
    sub_title += '\ncorrelation: ' + str(correlation)
    if not (subtitle is None):  sub_title = subtitle + '\n' + sub_title
    plt.figtext(0.6, 0.2, sub_title, ha='left')
    
    # Add mean of binvar and score
    rangexy = max_limit - min_limit
    MeansTitle = 'Actual ' + np.str(round(binary_variable.mean(), 3))
    MeansTitle = MeansTitle + '\nScore ' + np.str(round(score_variable.mean(), 3))
    plt.annotate(MeansTitle, xy=[min_limit + 0.1 * rangexy, max_limit - 0.1 * rangexy])
    plt.show()
    if wait:
        plt.waitforbuttonpress()
        plt.close()

    return


