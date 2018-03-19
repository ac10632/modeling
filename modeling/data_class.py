import numpy as np
import pandas as pd
from modeling.functions import linear_splines_basis1, linear_splines_basis2, categorical_to_design


class DataError(Exception):
    """
    Raised if there's a problem in the data
    
    """
    pass


class ModelError(Exception):
    """
    Raised if there's a problem in the model build

    """
    pass


class DataClass(object):
    """
    
    :param df: data from which to form the design matrix, dependent variable vector.
    :type df: pandas DataFrame
    
    The DataClass handles creating the design matrix and the dependent variable vector based
    on an R-type formula.
    
    DataClass supports these special features in the *formula*:
    
    - linear splines - basis 1 (hats).  The format:
    
        .. function:: h(var, knots, omit = None)
        
        - *var*: the name of the variable to form hats from.
        - *knots*: a *tuple* of knot points
        - *omit*: the (optional) basis function to omit from the design matrix.  Basis functions
          are numbered starting with 0.
        - See :meth:`functions.linear_splines_basis1`  for details of the basis functions.
    
    - linear splines - basis 2.  The format:
    
        .. function:: l(var, knots)
        
        - *var*: the name of the variable to form hats from.
        - *knots*: a *tuple* of knot points.
        - These basis functions automatically drop the constant term.
        - See :meth:`functions.linear_splines_basis2` for details of the basis functions.
    
    
    - categorical variables.  The format for creating a set of indicator variables in the
      design matrix for a categorical variable is:
      
        .. function:: c(var, omit = None)
        
        - *var*: the name of the variable in the DataFrame df to form the indicators from.
        - *omit*: the (optional) value of *var* to omit from the indicators to avoid
          collinearity with the intercept term in the model.
    
    Example usage:
    ::

        import pandas as pd
        import numpy as np
        from modeling_tools.data_class import DataClass
        n = 30000
        df = pd.DataFrame(np.random.uniform(0, 1, (n, 6)), columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x 6'])
        df['yr'] = np.random.randint(2000, 2017, n)
        y = -1 + 2 * df['x1'] - 3 * df['x2'] + 1.5*df['x3'] - .1*df['x4'] + 7.7*df['x5']
        df['ycts'] = y + np.random.normal(0, 1, n)
        design = DataClass(df)
        # Note variable names with internal spaces are OK.
        design.formula = 'ycts~ x1 + x2 + x3 + x4 + x5 + x 6'
        design.build_design()
        print ('There are this many observations ' + str(design.n))
        print ('There are this many parameters ' + str(design.p))
    
    
    """
    
    def __init__(self, df):
        self.__x = None
        self.__y = None
        self.__formula = ''
        self.__xColumnNames = None
        self.__event = None
        self.__wts = None
        self.__family = None
        self.__formula = None
        self.__has_intercept = True
        self.__event = None
        self.__implementationSkeleton = None
        self.__factors = {}
        if str(type(df)).find('pandas.core.frame.DataFrame') >= 0:
            self.df = df
        else:
            raise TypeError('DataClass: argument must be a Pandas DataFrame')
    
    @property
    def implementation_skeleton(self):
        """
        This list is a skeleton of an implementation of the model.  It is a list of lists. Each
        element of the list has the form:

        [*Python statement*, *parameter*]

        where *python statement* is a stub of a python statement associated with *parameter*.

        :Example: ['*fn = fn + x1*','x1']. So one only needs to add the value of the parameter
            on *x1* to complete the Python statement.

        Statements which don't need a parameter added to the end have *None*
        as the second entry.

        :return: implement elements
        :rtype: list


        """
        return self.__implementationSkeleton
    
    @property
    def factors(self):
        """

        Dictionary of factors in the model.  The keys are the variable names.  Each entry is also a dictionary with
        these elements:
        
            - type.  This is the type of the factor.
                
                - DIRECT.  The factor is in the model as-is (e.g. X1) without transformation (splines) or as a category.
                - SPLINES: HATS. The factor is in the model using linear splines basis 1.
                - SPLINES: LINEAR. The factor is in the model using linear splines basis 2.
                - CATEGORICAL. The factor is in the model as categorical.
                
            - knots.  numpy array of knots for splines
            
            - levels. For categorical variables, numpy array of levels of the variable.
            
            - omit. For splines basis 1, the basis function to drop.  For categorical variables, the level to omit.
              A value of None is permitted.
        
        :return: dictionary of factors
        :rtype: dict
        
        """
        return self.__factors
    
    @property
    def n(self):
        """
        Returns the number of rows in the design matrix if build_design() has been run.
        Otherwise *None* is returned.

        Note: cannot be set.

        :return: number of rows in the design matrix.
        :rtype: int


        """
        if self.x is not None:
            return self.x.shape[0]
        else:
            return None
    
    @property
    def p(self):
        """
        Returns the number of columns in the design matrix if build_design() has been run.
        Otherwise *None* is returned.

        This is the number of parameters to be estimated.

        Note: cannot be set.

        :return: number of columns in the design matrix.
        :rtype: int


        """
        if self.x is not None:
            return self.x.shape[1]
        else:
            return None
    
    @property
    def model_degrees_freedom(self):
        """
        Returns the degrees of freedom in the model if build_design() has been run.
        Otherwise *None* is returned.

        This is *n* - *p*.

        Note: cannot be set.

        :return: number of columns in the design matrix.
        :rtype: int


        """
        if self.x is not None:
            return self.x.shape[0] - self.x.shape[1]
        else:
            return None
    
    @property
    def event(self):
        """
        Logistic regression assumes that the response variable is binary 0/1 and that the modeled event is 1.

        The user may override this by specifying the event to be modeled.  In this way:
        
        - Any two-valued outcome (even strings) can be readily modeled without preprocessing.
        - If the data is already binary, the event 0 can be modeled.

        Returns *None* for Normal family or if unset by the user.

        :param value: The value of the dependent variable to be modeled as the 'event' in logistic regression
        :type value: int, str, ... same type as the dependent variable.
        :return: the value set by the user or *None* if unset.
        :rtype: same type as dependent variable.


        """
        return self.__event
    
    @event.setter
    def event(self, value):
        if value is None:
            self.__event = None
        else:
            if self.family == 'BINOMIAL':
                self.__event = value
            else:
                if self.family == 'NORMAL':
                    raise ModelError('DataClass: event is not meaningful for family Normal')
                else:
                    raise ModelError('DataClass: must specify family prior to event')
    
    @property
    def has_intercept(self):
        """

        :return: whether the model has an intercept
        :rtype: bool


        """
        return self.__has_intercept
    
    @property
    def formula(self):
        """
        Regression formula use to create design matrix and dependent variable vector.

        The formula has the form:
         <dependent variable> ~ <independent variable>_1+...+<independent variable>_k

        Special cases:

        - Start with '-1' to omit the intercept from the model.

        - .. function:: c(x_i,omit=None)
        
            treats *x_i* as a class variable.  *omit* is the optional value of x_i to omit from the
            design matrix.
          
        - .. function:: h(x_i,knots, omit=None)

            creates a set of linear splines based on *x_i*.  The knots used are specified in the tuple
            *knots*. The number of basis functions is equal to the number of knots. *omit* is the
            optional basis function to omit.  The basis functions are numbered starting with 0.

        - .. function:: l(x_i,knots)

            creates a set of linear splines based on *x_i*.  The knots used are specified in the tuple
            *knots*. The number of basis functions is equal to the number of knots. There is no option to
            omit a basis function.  The constant term is automatically dropped.

        :Example:
        
        *size ~ height + weight* fits a model of *size* to the factors *height* and *weight*.

        :Example:

        .. function:: c(state,'NY')
        
            creates a set of indicator values, one for each level of state.  The indicator corresponding
            to NY is omitted from the return DataFrame. Assuming state has 50 levels, the return DataFrame
            has 49 columns.  Their names are *state:AK*, *state:AL*, and so on.

        :Example:

        .. function:: h(height,(48, 60, 66, 72, 78),0)
        
        calculates linear spline basis 1 functions
        from the knot points 48, 60, 66, 72, 78.  The first basis function (0) is omitted.  The return
        DataFrame will have 4 columns.  Their names are: height1, height2, height3 and height4.

        :param value: regression formula
        :type value: str
        :return: formula specified by the user
        :rtype: str


        """
        return self.__formula
    
    @formula.setter
    def formula(self, value):
        self.__formula = value
    
    @property
    def wts(self):
        """
        Weights to use in the regression. Weights are optional and are permitted only for the Normal family.

        Requirements:
        
        - There must be *n* elements in the wts array.
        - Weights must be non-negative.

        :param value: wts to use in the regression
        :type value: str (column in df), numpy array, numpy vector
        :return: regression weights
        :rtype: numpy vector


        """
        return self.__wts
    
    @wts.setter
    def wts(self, value):
        
        if str(type(value)).find('numpy.ndarray') >= 0:
            value = np.matrix(value).T
        else:
            if str(type(value)).find('numpy.matrixlib.defmatrix.matrix') >= 0:
                if value.shape[0] == 1:
                    value = value.T
            else:
                if str(type(value)).find('str') > 0:
                    if (self.df.columns == value).sum() == 1:
                        value = np.matrix(self.df[value]).T
                    else:
                        raise ModelError('DataClass: weights not found in DataFrame')
                else:
                    if str(type(value)).find('pandas.core.series.Series') > 0:
                        value = np.matrix(value).T
                    else:
                        raise TypeError('DataClass: weights argument must be in ' +
                                        'input DataFrame or numpy array/matrix or Pandas series')
        
        self.__wts = value
        if (value < 0).any():
            raise TypeError('DataClass: weights must be non-negative')
        if self.__wts.shape[0] != self.df.shape[0]:
            raise ModelError('DataClass: weights must have same number of elements as input DataFrame')
    
    @property
    def x_column_names(self):
        """

        :return: names of the columns in the design matrix
        :rtype: pandas index numpy array


        """
        
        return self.__xColumnNames
    
    @property
    def family(self):
        """
        Regression distribution family.  The options are:

        - Normal.  This is standard linear regression. The dependent variable is normally distributed.

        - Binomial. This is logistic regression.  The dependent variable is binary.

        :return: distributional family to be used in the regression.
        :rtype: str


        """
        return self.__family
    
    @family.setter
    def family(self, value):
        v = value.upper()
        if v == 'NORMAL' or v == 'BINOMIAL':
            self.__family = v
        else:
            raise ModelError('DataClass: unsupported family name')
    
    @property
    def x(self):
        """
        The design matrix built upon the input DataFrame according to the formula.

        The matrix *x* is created by a call to build_design()

        :return: design matrix
        :rtype: numpy matrix


        """
        return self.__x
    
    @property
    def y(self):
        """
        The vector of the dependent variable built upon the input DataFrame
        as specified by the formula.

        Note: for logistic regression *y* is set up as binary and the value modeled is 1.

        :return: dependent variable
        :rtype: numpy vector


        """
        return self.__y
    
    def __parse_factor(self, factor):
        """
         Parses an individual element of a model specification, creates the needed columns and returns them.
         Input:

         Outputs:
           A dictionary of returns, most notable is 'df' which is a DataFrame of the columns for the design
           matrix that correspond to 'factor' input.
            df                   DataFrame      Output of the routine.
            varn                 string         Variable in the DataFrame to be worked on

            if the request is for linear splines (hats), the returned dictionary includes
            knots                 numpy array of knot points
            omit                  hat  that is omitted (e.g. 0=first hat, 1=second...) (may be None)

           if the request is for a class variable the returned dictionary includes
           base                  Level of the variable to be omitted (may be None)

           self.__implementationSkeleton   List       Used in implementing the model.

        :param factor: An individual element of self.formula
        :type factor: str
        :return: DataFrame of design matrix element corresponding to *factor* with additional information.
        :rtype: dict


        """
        factor = factor.strip()  # eliminate leading/trailing white space
        if factor == '':  # Nothing to do
            raise ModelError('DataClass: no factor specified while building design matrix')
        c = self.df.columns
        # case 1: this is variable in self.df
        if (c == factor).sum() == 1:
            self.__implementationSkeleton += [['fn += df_in["' + factor + '"] * ', factor]]
            return {'varn': factor, 'df': self.df[factor], 'type': 'DIRECT', 'knots': None, 'omit': None, 'levels': None}
        # case 2: hats/linear splines
        if (factor.find('h(') >= 0) or (factor.find('l(') >= 0):
            if factor.find('h(') >= 0:
                fn = 'HATS'
            else:
                fn = 'LINEAR'
            f1 = factor[2:len(factor)]  # drop 'h('
            i = f1.find(',')  # find variable name
            if i > 0:
                varn = f1[0:i]
                if (self.df.columns == varn).sum() == 0:
                    raise ModelError('DataCLass: factor ' + varn + ' not in DataFrame')
                f1 = f1[i:len(f1)]  # build knots array
                i1 = f1.find('(')
                i2 = f1.find(')')
                if i1 > 0 and i2 > 0 and i2 > i1:
                    i1 += 1
                    arr = f1[i1:i2]
                    f1 = f1[(i2 + 1):len(f1)]
                    ix = f1.find(',')  # see if there is a hat to omit from the return DataFrame
                    if ix >= 0:
                        omit = f1[(ix + 1):(len(f1) - 1)]
                        try:
                            omit = int(omit)
                        except ValueError:
                            raise ModelError('DataClass: knot to drop not a number')
                    else:
                        omit = None
                    # Build knots array
                    knots = []
                    i = arr.find(',')
                    while i > 0:
                        num = arr[0:i]
                        try:
                            n = float(num)
                        except ValueError:
                            raise ModelError('DataClass: knot point ' + num + ' not a number')
                        else:
                            knots += [n]
                            arr = arr[(i + 1):len(arr)]
                            i = arr.find(',')
                    try:
                        n = float(arr)
                    except ValueError:
                        raise ModelError('DataClass: knot point ' + arr + ' not a number')
                    else:
                        knots += [n]
                        knots = np.array(knots)  # now have knots as np array
                        if len(knots) <= 1:
                            raise ModelError('DataClass: must have at least 2 knot points')
                        else:
                            # Is omit a legal value?
                            if omit is not None:
                                if omit < 0 or omit >= knots.size:
                                    raise ModelError('DataClass: knot to drop out of range')
                            if fn == 'HATS':
                                df_out = linear_splines_basis1(self.df[varn], knots, omit)
                            else:
                                if omit is not None:
                                    raise ModelError(
                                        'DataClass: linear splines (basis 2) do not take basis to drop')
                                omit = 0
                                df_out = linear_splines_basis2(self.df[varn], knots, omit)
                            str_knots = '[' + str(knots[0])
                            # for implementation: create the call to hats
                            for elem in knots[range(1, knots.size)]:
                                str_knots += ', ' + str(elem)
                            str_knots += ']'
                            if fn == 'HATS':
                                hcall = 'xyz = linear_splines_basis1(df_in["' + varn + '"], ' + str_knots + ', ' + str(
                                    omit) + ')'
                            else:
                                hcall = 'xyz = linear_splines_basis2(df_in["' + varn + '"], ' + str_knots + ', ' + str(
                                    omit) + ')'
                            self.__implementationSkeleton += [[hcall, None]]
                            for col in df_out.columns:
                                self.__implementationSkeleton += [['fn += ' + 'xyz["' + col + '"] * ', col]]
                            return {'varn': varn, 'knots': knots, 'omit': omit, 'df': df_out, 'type': 'SPLINES: ' + fn, 'levels': None}
                else:
                    raise ModelError('DataClass: must have at least 2 knot points')
            else:
                raise ModelError('DataClass: must have knot points')
        # case 3: categorical variable
        if factor.find('c(') >= 0:
            f1 = factor[2:len(factor)]  # drop 'c('
            i = f1.find(',')  # there may or may not be a value to use as the base
            if i < 0:
                i = f1.find(')')
            if i > 0:
                varn = f1[0:i]
                if (self.df.columns == varn).sum() == 0:
                    raise DataError('DataClass: factor ' + varn + ' not in DataFrame')
                # look for level to omit
                f1 = f1[(i + 1):len(f1)]
                if f1 == '':
                    omit = None
                else:
                    omit = f1[0:(len(f1) - 1)]
                df_out = categorical_to_design(self.df[varn], omit)
                ccall = 'xyz = categorical_to_design(df_in["' + varn + '"], ' + '"' + omit + '", error_out=False)'
                self.__implementationSkeleton += [[ccall, None]]
                for col in df_out['df_out'].columns:
                    self.__implementationSkeleton += [['fn += ' + 'xyz["df_out"]["' + col + '"] * ', col]]
                return {'varn': varn, 'omit': omit, 'df': df_out['df_out'], 'type': 'CATEGORICAL', 'knots': None, 'levels': df_out['levels']}
            else:
                raise ModelError('DataClass: category not specified correctly: no variable name or missing , or )')
        # none of the above
        raise ModelError('DataClass: unknown error')
    
    def build_design(self):
        """
        Creates the design matrix and the dependent-variable vector.

        Populates these properties:

        - x
        - y
        - x_column_names
        - has_intercept
        - implementation_skeleton

        :Note: must set formula before calling build_design()

        :return: There is no direct return
        :rtype: N/A


        """
        if self.formula is None:
            raise DataError('must specify formula first')
        formula = self.formula
        # see if there is a dependent variable (this is optional...predict method doesn't need one
        i = self.formula.find('~')
        if i > 0:
            yvar = formula[0:i].strip()
            formula = formula[(i + 1):len(formula)]
            if (self.df.columns == yvar).sum() == 0:
                raise ModelError('DataClass: dependent variable not in DataFrame')
            self.__y = np.matrix(self.df[yvar].as_matrix()).T
            #
            if self.family == 'BINOMIAL':
                # check 2 values:
                yvals = np.unique(np.squeeze(np.array(self.__y)))
                if yvals.shape[0] != 2:
                    raise ModelError('DataClass: binomial dependent variable can have only two values')
                    # now check the actual values
                if self.event is not None:
                    if yvals[0] == self.event or yvals[1] == self.event:
                        y0 = np.zeros(self.__y.shape[0])
                        y1 = np.zeros(self.__y.shape[0])
                        y1.fill(1)
                        y = np.where(np.squeeze(np.array(self.__y)) == self.event, y1, y0)
                        y = np.matrix(y).T
                        self.__y = y
                    else:
                        raise ModelError(
                            'DataClass: binomial dependent variable value of event not in dependent variable')
                else:
                    if yvals[0] != 0 or yvals[1] != 1:
                        raise ModelError(
                            'DataClass: binomial dependent variable values must be 0 or 1 or event must be specified')
        
        r = formula.split('+')
        # Is there an intercept? Assume yes unless told no
        if r[0] == '-1':
            self.__implementationSkeleton = [['fn = 0', None]]
            self.__has_intercept = False
            df_out = None
            if len(r) == 1:
                raise ModelError('DataClass: no factors')
            r = r[1:len(r)]
        else:
            df_out = pd.DataFrame(np.full((self.df.shape[0], 1), 1), dtype='float64')
            df_out.columns = ['intercept']
        self.__implementationSkeleton = [['fn = ', 'intercept']]
        # Build up the rest of the matrix one factor at a time
        if r[0] != '':
            for factor in r:
                df_factor = self.__parse_factor(factor)
                if df_out is None:
                    df_out = pd.DataFrame(df_factor['df'])
                else:
                    df_out = df_out.join(df_factor['df'])
                self.__factors[df_factor['varn']] = {'type': df_factor['type'], 'knots': df_factor['knots'], 'omit': df_factor['omit'],
                                                     'levels': df_factor['levels']}
        if df_out is None:
            raise ModelError('DataClass: no factors specified')
        self.__x = np.matrix(df_out.as_matrix())
        self.__xColumnNames = df_out.columns
