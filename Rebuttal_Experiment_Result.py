'TiSASRec unfair': 
    accuracy(↑): [0.081, 0.0976, 0.1089, 0.1315, 0.2077, 0.2713, 0.3194, 0.4013, 0.4235, 0.4212]
    diversity(↑):[0.1123, 0.1132, 0.1122, 0.1128, 0.112, 0.1091, 0.1028, 0.0926, 0.0849, 0.0816] 
    fairness(↓): [0.0183, 0.0147, 0.0060, 0.0316, 0.0426, 0.053, 0.0368, 0.0157, 0.0005, 0.0034]
 'TiSASRec fair': 
    accuracy(↑):[0.0740, 0.0842, 0.1099, 0.1403, 0.1944, 0.2733, 0.3430, 0.3838, 0.4102, 0.4161]
    diversity(↑):[0.1100, 0.1114, 0.1116, 0.1128, 0.1119, 0.1067, 0.1025, 0.0902, 0.0832, 0.0812] 
    fairness(↓):[0.0084, 0.0029, 0.0024, 0.0091, 0.0131, 0.0044, 0.0100, 0.0183, 0.0227, 0.0194] 

'GRU4Rec unfair': 
    accuracy(↑):[0.1592, 0.1794, 0.1996, 0.2351, 0.2715, 0.2944, 0.3100, 0.3217, 0.3249, 0.3258]
    diversity(↑):[0.1021, 0.1018, 0.1017, 0.1006, 0.0986, 0.0965, 0.0946, 0.0926, 0.0910, 0.0895]
    fairness(↓):[0.0009, 0.0036, 0.0019, 0.0053, 0.0142, 0.0177, 0.0203, 0.0242, 0.0219, 0.0263]
'GRU4Rec fair': 
    accuracy(↑):[0.1570, 0.1742, 0.1982, 0.2360, 0.2723, 0.2918, 0.3092, 0.3178, 0.3251, 0.3254]
    diversity(↑):[0.1022, 0.1022, 0.1021, 0.1010, 0.0990, 0.0971, 0.0947, 0.0930, 0.0914, 0.0898] 
    fairness(↓):[0.0002, 0.0006, 0.0002, 0.0083, 0.0116, 0.0147, 0.0165, 0.0143, 0.0163, 0.0182]

When using a consistent backbone, there are still slight differences between the settings in Figure 6b and Figure 4a. This is because the analysis experiment introduces a fairness regularization term on top of the main experimental loss function, which alters the original form of the accuracy loss and thus introduces minor perturbations to accuracy. Specifically, we split the accuracy loss into two groups based on male and female users and increase the gap between their accuracy losses to reflect unfairness. The implementation can be found in models/BaseModel.py.
if self.is_fair == 0: # fair setting
    fairness_reg = 0.1 * max(male_acc_loss_mean, female_acc_loss_mean) + 0.9 * min(female_acc_loss_mean, male_acc_loss_mean)
else: # unfair setting
    fairness_reg = 0.9 * max(male_acc_loss_mean, female_acc_loss_mean) + 0.1 * min(female_acc_loss_mean, male_acc_loss_mean)

The conclusion is consistent with Figure 6: under fairness control, the fairness in the fair setting outperforms that in the unfair setting, while the differences in accuracy and diversity remain minimal.