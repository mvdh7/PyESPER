import numpy as np

def PyESPER_NN(X): #ESPER_silicate_1_Atl_1

    x1_step1 = {'xoffset':  [[-0.999999996490807], [-0.999999994516886], [-40], [0], [9.1619], [-1.99149636107278], [-0.03], [-177.232103709389], [-0.28]],
        'gain': [[1.0000000017546], [1.00003626070655], [0.0153846153846154], [0.000309885342423303], [0.0656325902039552], [0.0604107502911417], [0.604229607250755], [0.00425400504498441], [0.0470499670650231]],
        'ymin': -1}

    b1 = [[-0.78024273862288839876], [4.6172806491338240775], [-2.0480148021460116148], [0.0032291016185178062631], [3.8256647473996063802], [-1.6877461789895349398], [3.1396391191929899911], [3.4255673580556247337], [5.7603919532406155923], [5.1681729902937991739], [0.68282983932619090162], [1.8692667161381424368], [-7.3813584513138694732], [-6.8081686915677330774], [-2.8456047277512213078], [4.5415749914280274524], [-2.6644960318954247214], [-1.1442456699880336757], [2.5235155511807096929], [-0.92792967238335644087], [0.48869649252119468086], [-0.84505475982754596487], [-0.54510416199771472012], [5.1415048540684002987], [-2.2166224589125680389], [-1.6736459976572808372], [-2.5503177159043937117], [5.0294801548502254818], [5.8665735432428203211], [1.9053267476527444568], [-5.345561257074998629], [-0.20982859277924295616], [-1.6495491552383543254], [-2.635967885346020001], [-2.1231504678250940898], [3.3690359932009075195], [-8.1156079220321348799], [5.0829371633881024195], [-7.1646371475122210271], [0.80414370898782450148]]
    IW1_1 = [[0.51158677987348533112, -0.43695362115942726344, -0.50418799431949523626, -0.76193954737600200211, 1.68081478052323674, -0.75294469070018610335, -0.53253971724283588607, -0.23439388313043890699, 0.5662876366942123374], [0.44556800846630700841, 0.035297774631962923475, 2.5750611624862593807, -0.42480309362055751476, 3.7961495216476777159, 6.5492744219788496807, -1.3566880754726986513, 2.6811386269987798059, -1.5681605093044639343], [0.42740215236233264351, -0.30990373471072468758, 0.34914667381674202584, -0.59650710651104088544, 0.72447719763669227167, -0.73390483821843555123, 0.046258916405052329102, 0.37817291556906124095, 0.29680805279769467697], [-0.018442718188735253426, 0.098912885552392249, -0.96997264734050536727, 0.11763192858768582727, 0.6218025460517305758, -0.45771197089214044063, -0.32915973001608422877, -0.49745027535076652425, -0.029989572852542563386], [-0.093591036274865313516, 0.0904839230787014559, -0.4166508381593589383, 0.074912216498870698445, -0.31064369626141297154, 2.6524576185590245103, 0.19549235765383518593, 0.8229288579803690773, 0.81853016223516661398], [-0.31514971637024741247, -0.063308475201857075665, 0.7953843010749618303, -1.0223366192005767239, 1.7925083765033382743, 0.61784060140641283709, -0.79703321465031529147, -1.2525475133786563298, -0.50782448915951072799], [-3.0800618395120760162, 1.081642372833776955, 2.0101209762537770587, 0.32054556600900740992, 0.13336046080492353072, -0.75794411660569127598, -1.1117387276950903008, 0.98083089530981404369, 0.97163897688013922682], [-0.06806246127249192257, 0.080073746315000873808, -0.33712414693002673571, -0.07793454297156351962, -0.31684188182184341853, 2.57563163341399326, 0.33580768453254278283, 0.56995841281470738249, 0.64728807896589779514], [-0.77835439390211780264, -2.2085382636354506936, -2.1758812283598900628, 2.1426429369913866374, -6.0772198847691125678, -1.338261604779693803, 3.024200082765179598, -1.2862360204061409341, -3.0980396546342068298], [-0.86044270663945232602, -0.16968798409851412745, 0.53242037078657478144, -1.1195507300315994392, -4.7184367795533823653, 2.077679492553493823, -0.097259833447770366321, -0.72095649278010365268, -0.1899012997319657059], [-0.34174004361242332761, -0.32040184382821668141, -2.5352162727683977828, -0.47743160421950736616, -1.8700723280432312023, -0.83559245986448926757, -2.1272014642249676974, -1.5158704392688344686, 1.6175270654981295237], [-0.42143274196389296504, 0.25808926632445372551, -0.31279669870120868636, 0.40454458220128847179, -0.73323193913059514504, 0.91032593674572614972, -0.022620697907588388498, -0.36569562780202596874, -0.3266768325635070469], [-0.049171146711613628044, -0.18210842153910986041, -0.28713786351889253323, 1.1546320017175055739, 2.9204731270004788968, -7.4289593345474083108, -1.8090814687805316652, 1.9929076770811091635, -0.63913963172322707518], [-0.081458994520271332807, 0.54026481795604563985, -0.45467909433730790969, -1.0886910552238531213, -3.3876436507909155615, -7.5876482933508828665, -2.8324839093140914414, -2.7763876827828619653, -3.6712603005237469667], [2.2786034089371396405, -0.85960557098698764378, -0.81825699895174397458, 1.7942218997950507564, 0.88485245507016774269, 0.95842075232880341584, -0.42798596865759314101, -0.67806067644264644834, -0.83036544679231005794], [-0.71636236425327270982, -0.19543875582748448938, 0.67195874641777708636, -1.1137604245687733329, -4.0786722987506118798, 1.9614539190980249117, -0.16015283580367409288, -0.47970338688626595891, -0.1428016694044118573], [0.50396659114976261051, -0.13961419471416297577, 0.23777907304153625412, 0.82596540830337972228, -1.6919294798884070286, -3.7355464303324774633, 0.22200171431735721272, 1.499265289608793017, 0.17785617499460723567], [2.641453343357386796, -0.56931883720739495658, -3.3092386779255202889, 3.266219102120959672, -0.64549210420128910748, 1.0482712834402443391, 0.29863940569295333027, -2.0698879612893206215, -0.15271483152404621841], [0.73810126765460681852, 0.16128856742376693201, -0.70140473108623413445, -1.8959399566043999563, 0.73664890300687235758, 1.5962332901523155204, 2.9579430352054285613, 0.15213334906874995123, 0.33439860859529796366], [0.55429133560806387315, -2.0598694660440899362, 0.50411362424872729893, 0.73735351434661466907, -1.8178006409242555286, -0.36048065308227578152, -3.0259991995313093582, -1.1423345091333960699, -0.22606529761699395431], [-0.6958228465521377748, -0.053500433709240985403, -0.11241508826376392371, -1.1474188463517220882, -2.6755591205248294706, -0.20430056919539921201, 0.77625071114673493966, 1.3308781548123043148, 0.36226145248977920099], [0.96898087995264148287, 2.396809932069080773, 2.7683753796585297557, 0.81596534455439351241, 4.2220099881815915666, 0.98250893064300204305, -1.799695107841938535, 4.5354999172799193019, -0.058489320750204465416], [0.34896571861875452791, 0.38444186459454182669, 2.7353921132410579098, 0.48549686835138233798, 1.7923559190251714401, 0.94145571297717023374, 2.3851772066267393768, 1.679515450761430273, -1.8254195717685990363], [-0.39535519067958857509, -0.012003310417183513104, -0.67364407151874949875, 0.74156950130956411638, 0.25151336319810535525, 7.1486591525539164849, -2.1053417539345140597, -0.35992804155987634385, 2.4405508161340194739], [-2.4272652915618944114, -0.32039840527298241168, 1.0967468750015465506, -2.2366801373685762933, 4.1448055293766898899, -1.0059556656081407588, 3.6098791843345461317, 0.86693548759192851083, 0.088625145107470881811], [1.3179432202395624518, 0.057771414246515613578, 0.4035036799001178176, 1.2782702491108171028, 2.830100290650588768, -0.27663829934171052516, -1.4650075631282093447, 2.7783140211359294014, 0.43324964754473832551], [-2.8176809944070928537, -6.1726940621039290136, 0.8857564162388036344, 0.64001338136268448586, 0.24555325305473926778, -0.80205012461251090627, -0.35386650231527461141, -3.5752431903161561166, 1.7666044482107496894], [0.66998645913155885356, -0.092334843445625577885, -3.5167032441725214476, 2.0163363794002306939, 3.2005343786662607819, -2.2473873489773024126, -2.5120139815172484177, -0.99926346745676752903, 4.7254028863075410172], [5.397230567102286436, -5.3948316164067122713, 8.363563376097008728, 1.6777786600712085718, 0.2142619229884722698, -0.7958860819484450122, 0.42963020760223663563, -2.6421391893969650155, -0.96361061262127123417], [-0.61383163652543104583, 1.5824235795548240446, 1.1393851496231122145, -0.6512923397802412584, -0.043023563391104718834, -0.40482111574574253243, 0.15370747857350802734, -1.5187011282528830591, -0.77018304185828889707], [-1.1006193752007307207, 0.21117376545196320237, -2.3917015893172224317, 0.1329560123215301759, 0.0034673029435623274608, -6.0463982404506948498, 1.2941242112769579808, -0.55657749117155130847, -0.79592320720382203625], [-0.44954615679235687686, -0.040381823266050831389, 1.0436757445543245471, -1.1996005349750822333, -0.15366186626982322738, 0.23337248852515357633, -1.1953022807955380191, -1.9925356631019877085, -0.48033225670005319508], [-4.2358070019939635387, -0.029261495780276852025, 3.281333053455680826, 4.3070847453366045698, 0.66402737112969723032, 3.8903707371829385764, -3.3569226260957680807, 2.2143283893793954675, -3.1904599266643867139], [-2.7928916012020397197, -6.1409972479303895909, 0.84320816026360911888, 0.67402105113875554476, 0.41790282250830984578, -0.8721946612224102946, -0.19901619531564190835, -3.6146881145699922833, 1.6059916654480226317], [0.55788570464356967982, -2.0756749750958154443, 0.53783476621107406679, 0.38944637635104684126, -0.87957038612527194132, -0.53903274743125251245, -2.6638059546565449054, -1.4116014776455232393, -0.66963447207268045336], [0.82593908135976956686, -0.18995313776054092991, 0.90778990930290115458, 0.17709724761102602075, 2.6859363638650122219, 0.0040310783732398977608, 0.19665404801416971892, 1.4546225310638980499, 2.8312374258335317734], [-0.86440669101455880341, 0.20169073601413048391, -1.9670487598866532863, 0.4565425741945169924, 3.3401161476711003218, -6.4839594928852513078, 0.39373640786422164917, -0.23086330027199897241, 0.73946604349098066233], [1.0054049743746975576, -0.14603066382381263422, 1.7538229824144926372, 0.82155730282851702206, 0.63170510349589925614, 3.513528946965124522, -1.048034295702715557, 1.0039031493277608131, 1.1586320379956323023], [-0.7624286374806718003, 0.13627459625185855963, -1.8604586496946045049, 0.33047375586954313986, 2.2452898229036493127, -6.1718606267114415331, 0.53658290832670962889, -0.26405005116963037315, 0.44303402525188317895], [-0.18391729582484656058, 0.2818581199319021291, -0.88909220296739999156, 0.52753367957075825156, 0.43510455249749013795, -0.33434279418427820252, -0.19921324032330553488, -0.45269080123693489215, -0.17101182265633138591]]

    b2 = 2.9950202713897238205
    LW2_1 = [[-0.88096413123488837016, 0.40041622863771925145, -4.3526064696210537974, 1.9474942052170942652, -2.2593040015013556854, 0.34602293912112819241, -0.075253556611368538354, 2.7950975246065703494, 0.029579108186876503644, -1.5143538265342082472, -1.210385860301332217, -3.8123578460085552067, 0.086933040327984850704, 0.020384613324155715502, -0.13380246623897570912, 1.9250256375331009639, 0.36721868801090418177, 0.04539013275682741988, -0.12796837439404920578, 0.16767375562669606381, 0.21938935756341845784, -0.040393007348291813663, -1.0395506051015594995, -0.049380027016446281785, 0.032070974098903186345, 0.066385306562746229653, 0.33245766044390168137, 0.19116095120063705814, -2.2421224737785423464, 0.14655870414429864046, -0.75464657945990620824, -0.31019170597566397829, -0.01248178232307652416, -0.33064222676138826928, -0.16048728839347803365, 0.41009923530510583589, -3.0511849292333379502, -1.2437772351967808504, 4.7028009415161786677, -2.1983685353400606033]]

    y1_step1 = {'ymin':  -1,
        'gain': 0.0145232735458572,
        'xoffset': -0.52}

    TS = len(X[0])
    if len(X) != 0:
        Q = len(X[0][0][0])
    else:
        Q = 0

    Y = [None] * TS

    def mapminmax_apply(x, settings={}):
        y = np.subtract(x, settings['xoffset'])
        y = np.multiply(y, settings['gain'])
        y = np.add(y, settings['ymin'])
        return y

    def tansig_apply(n):
        return 2 / (1 + np.exp(-2 * n)) - 1

    def mapminmax_reverse(y, settings={}):
        x = np.subtract(y, settings['ymin'])
        x = np.divide(x, settings['gain'])
        x = np.add(x, settings['xoffset'])
        return x

    Xp1 = mapminmax_apply(X[TS-1],x1_step1)

    a1 = tansig_apply(np.tile(b1, (1, Q)) + np.dot(IW1_1, Xp1)[:, 0])

    a2 = np.tile(b2, (1, Q)) + np.dot(LW2_1, a1)

    Y = mapminmax_reverse(a2,y1_step1)

    return Y
