# Import relevant data for H2
knn = pd.read_csv('./kNN_preds.csv', sep=';')
manual = pd.read_csv('./test_labels.csv', sep=';')
core = pd.read_csv('./CoreData.csv', sep=';')
depth = core[['UttEnum', 'Depth']]

# join pred and test data
x = knn.join(manual)
xidx = x.set_index('UttEnum')
didx = depth.set_index('UttEnum')
df = xidx.join(didx) #df provides pred, test, and depth data for each utterance


unq = list(df['Depth'].unique())
unq.sort()

corrs = list()
for d in unq:
	data = df[df['Depth'] == d]
	preds = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
	labels = data.iloc[:, [9, 10, 11, 12, 13, 14, 15, 16, 17]]
	corr = list()
	for r in range(len(data)):
		pred = list(preds.iloc[r])
		label = list(labels.iloc[r])
		corr.append(scipy.stats.spearmanr(pred, label, nan_policy='omit')[0])
	corrs.append(corr)
    
avg_corrs = list()
for c in corrs:
    avg_corrs.append(np.nanmean(c))

print("Depths: ")
print(unq)
print("Average Correlations: ")
print(avg_corrs)

linr = scipy.stats.linregress(avg_corrs, unq)
print(linr)