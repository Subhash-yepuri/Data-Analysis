
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''***Importing data***'''

data=pd.read_csv('''banking_data.csv''')

print(data.shape)

'''***data cleaning***'''

print(data.isnull().sum())

print(data.keys())

data.dropna(inplace=True)
print(data.isnull().sum())

print(data.shape)

print(data.head(10))

"""*** Distribution of Age among Clients ***"""

new1_data=data[data['y']=='yes']

plt.figure(figsize=(7,7))
plot=plt.hist(data['age'],bins=20,color='green',edgecolor='black')
new_plot=plt.hist(new1_data['age'],bins=20,color='#40DF40',edgecolor='black')
for bin in range(len(new_plot[0])-1):
  height=new_plot[0][bin]/plot[0][bin].sum()*100
  plt.text(new_plot[1][bin],new_plot[0][bin],f'{height:0.0f}',ha='left',va='bottom',fontdict={'fontsize':10,'color':'black'})
plt.ylabel('count')
plt.title('age distribution')
x_ticks = np.arange(min(data['age']), max(data['age']) + 1, 4)
plt.xticks(x_ticks)
y_ticks = np.arange(0, max(plot[0])+1, 300)
plt.yticks(y_ticks)
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_age_among_clients.png',dpi=300,bbox_inches='tight')
plt.show()

"""***Job variation among Clients***"""

print(data['job'].value_counts())

plt.figure(figsize=(7,7))
colours=sns.color_palette('pastel')[0:len(data['job'].value_counts())]
explode=[(0.1) for i in range(len(data['job'].value_counts()))]
plt.pie(data['job'].value_counts(),labels=data['job'].value_counts().index,shadow=True,autopct='%0.2f%%',explode=explode,colors=colours)
plt.xlabel('job')
plt.ylabel('count')
plt.title('job variation')
#if you want to download the image uncomment the following line
# plt.savefig('Job_variation_among_clients.png',dpi=300,bbox_inches='tight')
plt.show()

plt.figure(figsize=(8,8))
colours=sns.color_palette('pastel')[0:len(data['job'].value_counts())]
colours1=sns.color_palette('colorblind')[0:len(data['job'].value_counts())]
bar_positions=np.arange(len(data['job'].value_counts()))*1.5
bars=plt.bar(bar_positions,data['job'].value_counts(),color=colours)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')
yes_job=[]
for i in data['job'].value_counts().index:
    cond1=data['job']==i
    cond2=data['y']=='yes'
    job_yes=cond1&cond2
    job_yes_count=job_yes.sum()
    yes_job.append(job_yes_count)
bars1=plt.bar(bar_positions,yes_job,color=colours1)
for i in range(len(bars1)):
  bar=bars1[i]
  percentage=(yes_job[i]/(data['job'].value_counts()[i]))*100
  height=bar.get_height()
  plt.text(bar.get_x()+bar.get_width()/2.0,height,f'yes:{yes_job[i]} \n percentage:{percentage:0.1f}',ha='center',va='bottom',fontdict={'fontsize':5})
plt.xlabel('job',fontdict={'fontname': 'serif', 'fontsize': 14, 'color': 'black'})
plt.ylabel('count',fontdict={'fontname': 'Serif', 'fontsize': 14, 'color': 'black'})
plt.xticks(bar_positions,data['job'].value_counts().index,rotation=90)
plt.title('job variation',fontdict={'fontname': 'Serif', 'fontsize': 16, 'color': 'black'})
#if you want to download the image uncomment the following line
#plt.savefig('Job_variation_among_clients_bar.png',dpi=300,bbox_inches='tight')

plt.show()

"""***Marital status distribution***"""

data['marital_stats1']=data['marital_status'].replace({'married':1,'single':0,'divorced':2})
data['marital_stats2']=data['marital'].replace({'married':1,'single':0,'divorced':2})
correlation=data['marital_stats1'].corr(data['marital_stats2'])
print(correlation)

data.drop(['marital_stats1','marital_stats2'],inplace=True,axis=1)

print(data['marital_status'].value_counts())

plt.figure(figsize=(6,6))
colours=sns.color_palette('colorblind')[0:len(data['marital_status'].value_counts())]
colours1=sns.color_palette('pastel')[0:len(data['marital_status'].value_counts())]
plt.xlabel('Marital Status',fontdict={'fontname':'serif','fontsize':'14','color':'black'})
plt.ylabel('Count',fontdict={'fontname':'serif','fontsize':'14','color':'black'})
plt.title('Marital Status Distribution',fontdict={'fontname':'serif','fontsize':'16','color':'black'})

married_yes = ((data['marital'] == 'married') & (data['y'] == 'yes'))
married_yes_count=married_yes.sum()
single_yes = ((data['marital'] == 'single') & (data['y'] == 'yes'))
single_yes_count=single_yes.sum()
divorced_yes = ((data['marital'] == 'divorced') & (data['y'] == 'yes'))
divorced_yes_count=divorced_yes.sum()

bars=plt.bar(data['marital_status'].value_counts().index,data['marital_status'].value_counts(),color=colours1)
bars1=plt.bar(data['marital_status'].value_counts().index,[married_yes_count,single_yes_count,divorced_yes_count],color=colours)
for bar in bars:
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2.0,height,f'{height}',ha='center',va='bottom')
for bar in bars1:
    i=0
    height=bar.get_height()
    percentage=(height/data['marital_status'].value_counts()[i])*100
    plt.text(bar.get_x()+bar.get_width()/2.0,height,f'yes:{height} \n percentage:{percentage:0.2f}',ha='center',va='bottom')
    i+=1
#if you want to download the image uncomment the following line
#plt.savefig('Marital_status_distribution.png',dpi=300,bbox_inches='tight')    
plt.show()

'''***What is the level of education among the clients?***'''

print(data['education'].value_counts())

cond1=data['education']=='secondary'
cond2=data['education']=='tertiary'
cond3=data['education']=='primary'
cond4=data['education']=='unknown'
cond5=data['y']=='yes'
cond6=data['y']=='no'
a1=(cond1 & cond5).sum()
a2=(cond2 & cond5).sum()
a3=(cond3 & cond5).sum()
a4=(cond4 & cond5).sum()
education_yes_counts=[a1,a2,a3,a4]
education_yes_counts_percentage=[(education_yes_counts[i]/data['education'].value_counts()[i])*100 for i in range(len(education_yes_counts))]
education_data=pd.DataFrame({'education':data['education'].value_counts().index,'Total':data['education'].value_counts().values,'yes':education_yes_counts,'percentage':education_yes_counts_percentage})
print(education_data)

'''***What proportion of clients have credit in default?***'''

print(data['default'].value_counts())
print('Proportion percentage of clients with credit in default:',(data['default'].value_counts()[1]/data['default'].value_counts().sum())*100)
print('Proportion percentage of clients without credit in drefault:',data['default'].value_counts()[0]/data['default'].value_counts().sum()*100)

"""***Distribution of average yearly balance among the clients***"""

average_yearly_balance = data['balance']
plt.figure(figsize=(10, 6))
sns.histplot(average_yearly_balance, bins=200, edgecolor='k',kde=True, alpha=0.7)
sns.histplot(new1_data['balance'],bins=200,kde=True,color='red')
plt.title('Distribution of Average Yearly Balance Among Clients')
plt.xlabel('Average Yearly Balance')
plt.ylabel('Number of Clients')
plt.xlim(-5000,40000)
xticks=np.arange(-5000,40000,2000)
plt.xticks(xticks,fontsize=6)
plt.grid(True)
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_average_yearly_balance.png',dpi=300,bbox_inches='tight')
plt.show()

"""***Clients having housing loans***"""

print('Clients having housing loans:',data['housing'].value_counts()['yes'])
cond1=data['housing']=='yes'
cond2=data['y']=='yes'
comb_cond=cond1 & cond2
housingyes_subscribed=data[comb_cond].shape[0]
print('clients having housing loan nSubscribed:',housingyes_subscribed)
cond1=data['housing']=='no'
cond2=data['y']=='yes'
comb_cond=cond1 & cond2
housingno_subscribed=data[comb_cond].shape[0]
print('clients having no housing loans subscribed:',housingno_subscribed)

"""***Clients having personal loans***"""

print('Clients having personal loans:',data['loan'].value_counts()['yes'])
cond1=data['loan']=='yes'
cond2=data['y']=='yes'
comb_cond=cond1 & cond2
loanyes_subscribed=data[comb_cond].shape[0]
print('clients having personal loan Subscribed:',loanyes_subscribed)
cond1=data['loan']=='no'
cond2=data['y']=='yes'
comb_cond=cond1 & cond2
loanno_subscribed=data[comb_cond].shape[0]
print('clients having no personal loans subscribed:',loanno_subscribed)

"""***Communication types used for contacting clients during the campaign***"""

print('Communication types used for contacting clients during the campaign:',[i for i in data['contact'].value_counts().index])

"""***Distribution of last contact day of the month***"""

plt.figure(figsize=(10,6))
sns.histplot(data['day'],bins=30,kde=True,color='red')
plt.hist(data['day'],alpha=0.6,bins=30,color='yellow',edgecolor='black')
plt.title('Distribution of last contact day of the month')
plt.xlabel('last contact day of the month')
xticks=np.arange(1,32,1)
plt.xticks(xticks)
yticks=np.arange(0,3000,200)
plt.yticks(yticks)
plt.ylabel('count')
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_last_contact_day_of_month.png',dpi=300,bbox_inches='tight')
plt.show()

print(data['poutcome'].value_counts())

plt.figure(figsize=(12,12))

'''***How does the last contact month vary among the clients?***'''

months_counts=[]
months_yes_counts=[]
months=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
for i in range(len(months)):
  cond1=data['month']==months[i]
  cond2=data['poutcome']=='success'
  months_yes=cond1&cond2
  months_yes_counts.append(months_yes.sum())
for i in range(len(months)):
  months_counts.append((data['month']==months[i]).sum())
a=plt.subplot(2,1,1)
main_bar=plt.bar(months,months_counts,alpha=0.6,color='#1CE8C0',edgecolor='#049074')

for bar in main_bar:
  height=bar.get_height()
  plt.text(bar.get_x()+bar.get_width()/2.0,height,f'{height}',ha='center',va='bottom')
bars=plt.bar(months,months_yes_counts,alpha=0.9,color='#066350',edgecolor='#049074')
for bar in bars:
  height=bar.get_height()
  plt.text(bar.get_x()+bar.get_width()/2.0,height,f'{height}',ha='center',va='center')
plt.legend(['Total count','Success after campaign'], loc='upper right')
plt.title('Distribution of last contact month')
plt.xlabel('Last contact month')
plt.ylabel('count')

percentage=np.array(months_yes_counts)/np.array(months_counts)*100

sorting=sorted(list(zip(percentage,months,months_counts)))

b=plt.subplot(2,1,2)

plt.bar([sorting[i][1] for i in range(len(months))],[sorting[i][2] for i in range(len(months))],alpha=0.6,color='#1CE8C0',edgecolor='#049074')
plt.bar([sorting[i][1] for i in range(len(months))],[sorting[i][0] for i in range(len(months))],alpha=0.9,color='#066350',edgecolor='#049074')
plt.title('Distribution according to success percentage')
plt.text(0.95, 0.95, 'Sorted in ascending order \nof success percentage',transform=b.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5))
plt.xlabel('Last contact month')
plt.ylabel('count')
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_last_contact_month.png',dpi=300,bbox_inches='tight')
plt.show()

'''***What is the distribution of the duration of the last contact?***'''

print(data['duration'].value_counts())
print("max duration:",max(data['duration']))
print("min duration:",min(data['duration']))

plt.figure(figsize=(10,6))
sns.histplot(data['duration'],bins=100,kde=True,color='red',alpha=0.7)
plt.hist(data['duration'],alpha=0.6,bins=100,color='yellow',edgecolor='black')
plt.title('Distribution of duration of the last contact')
plt.xlabel('Duration')
plt.ylabel('count')
x_ticks=np.arange(0,5000,200)
plt.xticks(x_ticks,fontsize=7)
y_ticks=np.arange(0,8000,500)
plt.yticks(y_ticks,fontsize=7)
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_duration_of_last_contact.png',dpi=300,bbox_inches='tight')
plt.show()

"""***contacts performed during the campaign for each client***"""

contacts_per_client = data.groupby(['job','marital','education'])['campaign'].sum().reset_index()
contacts_per_client.columns = ['job','marital','education', 'Total Contacts']
print(contacts_per_client.to_string())

'''***What is the distribution of the number of days passed since the client was last contacted from a previous campaign?***'''

a=(data['pdays'].value_counts())
print(a)
data_without_outliers = data[data['pdays'] != -1]

plt.figure(figsize=(14,7))
sns.histplot(data_without_outliers['pdays'],bins=50,kde=True,color='red',alpha=0.7)
a=plt.hist(data_without_outliers['pdays'],alpha=0.6,bins=50,color='yellow',edgecolor='black')
plt.title('Distribution of number of days that passed by after the client was last contacted from a previous campaign')
plt.xlabel('Number of days')
x_ticks=np.arange(0,900,50)

for i in range(len(a[1])-1):
  height=a[0][i]
  plt.text(a[1][i],height,f'{height:0.0f}',ha='left',va='bottom',fontdict={'fontsize':6})
plt.xticks(x_ticks,fontsize=7)
y_ticks=np.arange(0,1500,100)
plt.yticks(y_ticks,fontsize=7)
plt.ylabel('count')
#if you want to download the image uncomment the following line
# plt.savefig('Distribution_of_no_of_days__passed.png',dpi=300,bbox_inches='tight')
plt.show()

"""***Contacts performed before the current campaign for each client***"""

previous_contacts_for_each_client=data.groupby(['job','marital','education'])['previous'].sum().reset_index()
previous_contacts_for_each_client.columns=['job','marital','education','Total Previous Contacts']
print(previous_contacts_for_each_client.to_string())

'''***What were the outcomes of the previous marketing campaigns?***'''

print(data['poutcome'].value_counts())

'''***What is the distribution of clients who subscribed to a term deposit vs. those who did not?***'''

subscribed_to_term_deposit=data['y'].value_counts()['yes']
not_subscribed_to_term_deposit=data['y'].value_counts()['no']
print('No of people subscribed to term deposit:' ,subscribed_to_term_deposit)
print('No of people not subscribed to term deposit:' ,not_subscribed_to_term_deposit)

'''***Are there any correlations between different attributes and the likelihood of subscribing to a term deposit?***'''

import matplotlib.colors as mcolors
fig=plt.figure(figsize=(14,8))
data_of_subscribed_to_term_deposit=data[data['y']=='yes']
a=plt.hist(data_of_subscribed_to_term_deposit['age'],bins=30,edgecolor='black')
plt.clf()
norm = mcolors.Normalize(vmin=min(a[0]), vmax=max(a[0]))
cmap = plt.get_cmap('viridis')
bin_edges = a[1]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for count, left, right in zip(a[0], bin_edges[:-1], bin_edges[1:]):
    color = cmap(norm(count))
    plt.bar(left, count, width=right-left, color=color, edgecolor='black')
plt.xlabel('age')
plt.ylabel('count')
plt.title('Age distribution for people subscribed for term deposit')
x_ticks = np.arange(min(data['age']), max(data['age']) + 10, 5)
plt.xticks(x_ticks)
y_ticks = np.arange(0,1000, 200)
plt.yticks(y_ticks)
for i in range(len(a[0])):
  height=a[0][i]
  plt.text(a[1][i],height,f'{height:0.0f}',ha='center',va='bottom',fontdict={'fontsize':10})
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Count')
#if you want to download the image uncomment the following line
# plt.savefig('Age_distribution_of_subscribed_people.png',dpi=300,bbox_inches='tight')
plt.show()

fig=plt.figure(figsize=(14,8))
b=plt.hist(data_of_subscribed_to_term_deposit['balance'],bins=50,edgecolor='black')
plt.clf()
norm = mcolors.Normalize(vmin=min(b[0]), vmax=max(b[0]))
cmap = plt.get_cmap('viridis')
bin_edges = b[1]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for count, left, right in zip(b[0], bin_edges[:-1], bin_edges[1:]):
    color = cmap(norm(count))
    plt.bar(left, count, width=right-left, color=color, edgecolor='black')
plt.xlabel('balance')
plt.ylabel('count')
plt.title('balance distribution for people subscribed for term deposit')
x_ticks = np.arange(-5000, 90000, 5000)
plt.xticks(x_ticks)
y_ticks = np.arange(min(b[0]),max(b[0]), 200)
plt.yticks(y_ticks,fontsize=9)
for i in range(len(b[0])):
  height=b[0][i]
  plt.text(b[1][i],height,f'{height:0.0f}',ha='center',va='bottom',fontdict={'fontsize':10})
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Count')
#if you want to download the image uncomment the following line
#plt.savefig('balance_distribution_of_subscribed_people.png',dpi=300,bbox_inches='tight')
plt.show()

subscription_counts = data['y'].value_counts()

# Plot the distribution
plt.figure(figsize=(10, 6))
subscription_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Number of Clients')
plt.title('Distribution of Clients Subscribed vs. Not Subscribed to Term Deposit')
plt.xticks(rotation=0)
#if you want to download the image uncomment the following line
# plt.savefig('subscribed_vs_unsubscribed.png',dpi=300,bbox_inches='tight')
plt.show()

print(data.info())

print(data['job'].value_counts().index)
print(data['marital'].value_counts().index)
print(data['education'].value_counts().index)
print(data['default'].value_counts().index)
print(data['housing'].value_counts().index)
print(data['loan'].value_counts().index)
print(data['contact'].value_counts().index)
print(data['month'].value_counts().index)
print(data['day_month'].value_counts().index)
print(data['poutcome'].value_counts().index)
print(data['y'].value_counts().index)

new_data=data.copy()
new_data['job']=data['job'].replace(data['job'].value_counts().index,range(len(data['job'].value_counts().index)))
new_data['marital']=data['marital'].replace(data['marital'].value_counts().index,range(len(data['marital'].value_counts().index)))
new_data['education']=data['education'].replace(data['education'].value_counts().index,range(len(data['education'].value_counts().index)))
new_data['default']=data['default'].replace(data['default'].value_counts().index,range(len(data['default'].value_counts().index)))
new_data['housing']=data['housing'].replace({'no':0,'yes':1})
new_data['loan']=data['loan'].replace(data['loan'].value_counts().index,range(len(data['loan'].value_counts().index)))
new_data['contact']=data['contact'].replace(data['contact'].value_counts().index,range(len(data['contact'].value_counts().index)))
new_data['month']=data['month'].replace(data['month'].value_counts().index,range(len(data['month'].value_counts().index)))
new_data['day_month']=data['day_month'].replace(data['day_month'].value_counts().index,range(len(data['day_month'].value_counts().index)))
new_data['poutcome']=data['poutcome'].replace(data['poutcome'].value_counts().index,range(len(data['poutcome'].value_counts().index)))
new_data['y']=data['y'].replace(data['y'].value_counts().index,range(len(data['y'].value_counts().index)))
new_data['marital_status']=data['marital_status'].replace(data['marital_status'].value_counts().index,range(len(data['marital_status'].value_counts().index)))
print(new_data.head())

corr_matrix=new_data.corr()
print(corr_matrix)

for i in range(len(new_data.columns)):
  for j in range(i+1,len(new_data.columns)):
    if corr_matrix.iloc[i,j]>0.5 or corr_matrix.iloc[i,j]< -0.5:
      print(new_data.columns[i],new_data.columns[j])
      print(corr_matrix.iloc[i,j])

