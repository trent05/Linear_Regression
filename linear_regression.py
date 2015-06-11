import numpy as np
import statsmodels.api as sm
import pandas as pd
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
InterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = InterestRate
FICO_Score = loansData['FICO.Range'].map(lambda x: x.split('-'))
FICO_Score = FICO_Score.map(lambda x: [int(n) for n in x])
loansData['FICO.Range'] = FICO_Score
loansData['FICO.Score'] = loansData['FICO.Range']
def F_S(dt):
	return dt[0]
loansData['FICO.Score'] = loansData['FICO.Score'].apply(F_S)
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']
y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
x = np.column_stack([x1,x2])
x = sm.add_constant(x)
model = sm.OLS(y,x)
f = model.fit()
print f.summary()