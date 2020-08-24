from os import path
import numpy as np

from ntuple_processor.utils import Selection
from ntuple_processor.utils import Cut
from ntuple_processor.utils import Weight
from ntuple_processor.variations import ReplaceCut, AddWeight, ChangeDataset

# Base path to main ntuples
basepath = '/ceph/htautau/deeptau_02-20/2018/'
ntuples_base = path.join(basepath, 'ntuples')

# Friend trees
friends_base = [path.join(basepath, 'friends', f) for f in ['TauTriggers', 'SVFit']]
ml_score_base = ['/home/wunsch/workspace/mlnll-analysis/output/MLScores']

# File list
files = {
    'singlemuon': [
        'SingleMuon_Run2018A_17Sep2018v2_13TeV_MINIAOD',
        'SingleMuon_Run2018B_17Sep2018v1_13TeV_MINIAOD',
        'SingleMuon_Run2018C_17Sep2018v1_13TeV_MINIAOD',
        'SingleMuon_Run2018D_22Jan2019v2_13TeV_MINIAOD'
        ],
    'wjets': [
        'W1JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W2JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W3JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'W4JetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'WJetsToLNu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2'
        ],
    'dy': [
        'DY1JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY2JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY3JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DY4JetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v1',
        'DYJetsToLLM10to50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v2',
        'DYJetsToLLM50_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_madgraph-pythia8_v1'
        ],
    'tt': [
        'TTTo2L2Nu_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'TTToHadronic_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'TTToSemiLeptonic_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1'
        ],
    'vv': [
        'WW_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v2',
        'ZZ_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v2',
        'WZ_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_pythia8_v3',
        'STt-channelantitop4fInclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'STt-channeltop4fInclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'STtWantitop5finclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1',
        'STtWtop5finclusiveDecaysTuneCP5_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1'
        ],
    'ggh': [
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin101M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin104to105M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin106M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin107to109M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'GluGluHToTauTauHTXSFilterSTXS1p1Bin110to113M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'GluGluHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2'
        ],
    'qqh': [
        'VBFHToTauTauHTXSFilterSTXS1p1Bin203to205M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauHTXSFilterSTXS1p1Bin206M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauHTXSFilterSTXS1p1Bin207to210M125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v1',
        'VBFHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_ext1-v1',
        'ZHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'WminusHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2',
        'WplusHToTauTauM125_RunIIAutumn18MiniAOD_102X_13TeV_MINIAOD_powheg-pythia8_v2'
        ]
    }

# Selections

channel = Selection(name = 'mt',
               cuts = [
                       ('flagMETFilter == 1', 'METFilter'),
                       ('extraelec_veto<0.5', 'extraelec_veto'),
                       ('extramuon_veto<0.5', 'extramuon_veto'),
                       ('dilepton_veto<0.5', 'dilepton_veto'),
                       ('byTightDeepTau2017v2p1VSmu_2>0.5', 'againstMuonDiscriminator'),
                       ('byVVLooseDeepTau2017v2p1VSe_2>0.5', 'againstElectronDiscriminator'),
                       ('byTightDeepTau2017v2p1VSjet_2>0.5', 'tau_iso'),
                       ('iso_1<0.15', 'muon_iso'),
                       ('q_1*q_2<0', 'os'),
                       ('((pt_2>30) && ((trg_singlemuon_27 == 1) || (trg_singlemuon_24 == 1))) || ((pt_1<25) && (trg_crossmuon_mu20tau27_hps == 1 || trg_crossmuon_mu20tau27 == 1))', 'trg_selection')
               ])

# TODO: Add the correct trigger weight string (see below)
#triggerweight = '((trg_singlemuon_27 || trg_singlemuon_24)*((((pt_1>=25)&&(pt_1<28))*trigger_24_Weight_1)+((pt_1>=28)*(trigger_24_27_Weight_1)))+(pt_1 > 21 && pt_1 < 25 && trg_crossmuon_mu20tau27_hps)*(crossTriggerDataEfficiencyWeight_1*((byTightDeepTau2017v2p1VSjet_2<0.5 && byVLooseDeepTau2017v2p1VSjet_2>0.5)*crossTriggerCorrectedDataEfficiencyWeight_vloose_DeepTau_2 + (byTightDeepTau2017v2p1VSjet_2>0.5)*crossTriggerCorrectedDataEfficiencyWeight_tight_DeepTau_2))/(crossTriggerMCEfficiencyWeight_1*((byTightDeepTau2017v2p1VSjet_2<0.5 && byVLooseDeepTau2017v2p1VSjet_2>0.5)*crossTriggerCorrectedMCEfficiencyWeight_vloose_DeepTau_2 + (byTightDeepTau2017v2p1VSjet_2>0.5)*crossTriggerCorrectedMCEfficiencyWeight_tight_DeepTau_2)))'
triggerweight = '(crossTriggerMCWeight_1*(crossTriggerMCWeight_1<10 && crossTriggerMCWeight_1>0.1)+(crossTriggerMCWeight_1>10 || crossTriggerMCWeight_1<0.1))*(pt_1<25) + (trigger_24_27_Weight_1*(pt_1>25))'

mc = Selection(name = 'mc',
        weights = [
            ('generatorWeight', 'generatorWeight'),
            ('puweight', 'puweight'),
            ('idWeight_1*idWeight_2', 'idweight'),
            ('isoWeight_1*isoWeight_2', 'isoweight'),
            ('trackWeight_1*trackWeight_2', 'trackweight'),
            (triggerweight, 'triggerweight'),
            ('((gen_match_2 == 5)*tauIDScaleFactorWeight_tight_DeepTau2017v2p1VSjet_2+ (gen_match_2 != 5))', 'taubyIsoIdWeight'),
            ('59.7 * 1000.0', 'lumi')
            ]
        )

w = Selection(name = 'w',
        weights = [
            ('((0.00092600048*((npartons <= 0 || npartons >= 5)*1.0 + (npartons == 1)*0.1647043928 + (npartons == 2)*0.128547226623 + (npartons == 3)*0.0767138313139 + (npartons == 4)*0.0631529545476)) * (genbosonmass>=0.0) + numberGeneratedEventsWeight * crossSectionPerEventWeight * (genbosonmass<0.0))', 'wj_stitching_weight'),
            ('eleTauFakeRateWeight*muTauFakeRateWeight', 'leptonTauFakeRateWeight')
            ]
        )

dy = Selection(name = 'dy',
        weights = [
            ('((genbosonmass >= 50.0)*0.0000606542*((npartons == 0 || npartons >= 5)*1.0 + (npartons == 1)*0.194267667208 + (npartons == 2)*0.21727746547 + (npartons == 3)*0.26760465744 + (npartons == 4)*0.294078683662) + (genbosonmass < 50.0)*numberGeneratedEventsWeight*crossSectionPerEventWeight)', 'z_stitching_weight'),
            ('zPtReweightWeight', 'zPtReweightWeight')
            ]
        )

ztt = Selection(name = 'ztt',
        cuts = [
            ('gen_match_1==4 && gen_match_2==5', 'ztt_cut')
            ]
        )

zl = Selection(name = 'zl',
        cuts = [
            ('!(gen_match_1==4 && gen_match_2==5) && !(gen_match_2 == 6)', 'zl_cut')
            ]
        )

zj = Selection(name = 'zj',
        cuts = [
            ('gen_match_2 == 6', 'zj_cut')
            ]
        )

tt = Selection(name = 'tt',
        weights = [
            ('numberGeneratedEventsWeight', 'numberGeneratedEventsWeight'),
            ('crossSectionPerEventWeight', 'crossSectionPerEventWeight'),
            ('topPtReweightWeightRun2', 'topPtReweightWeight'),
            ('eleTauFakeRateWeight*muTauFakeRateWeight', 'leptonTauFakeRateWeight')
            ]
        )

ttt = Selection(name = 'ttt',
        cuts = [
            ('gen_match_1==4 && gen_match_2==5', 'ttt_cut')
            ]
        )

ttl = Selection(name = 'ttl',
        cuts = [
            ('!(gen_match_1==4 && gen_match_2==5) && !(gen_match_2 == 6)', 'ttl_cut')
            ]
        )

ttj = Selection(name = 'ttj',
        cuts = [
            ('gen_match_2 == 6', 'ttj_cut')
            ]
        )

vv = Selection(name = 'vv',
        weights = [
            ('numberGeneratedEventsWeight', 'numberGeneratedEventsWeight'),
            ('eleTauFakeRateWeight*muTauFakeRateWeight', 'leptonTauFakeRateWeight'),
            ('crossSectionPerEventWeight', 'crossSectionPerEventWeight')
            ]
        )

vvt = Selection(name = 'vvt',
        cuts = [
            ('gen_match_1==4 && gen_match_2==5', 'vvt_cut')
            ]
        )
vvl = Selection(name = 'vvl',
        cuts = [
            ('!(gen_match_1==4 && gen_match_2==5) && !(gen_match_2 == 6)', 'vvl_cut')
            ]
        )

vvj = Selection(name = 'vvj',
        cuts = [
            ('gen_match_2 == 6', 'vvj_cut')
            ]
        )

htt = Selection(name = 'htt',
        weights = [
            ('eleTauFakeRateWeight*muTauFakeRateWeight', 'leptonTauFakeRateWeight')
            ]
        )

ggh = Selection(name = 'ggh',
        weights = [
            ('ggh_NNLO_weight', 'gghNNLO'),
            ('1.01', 'bbh_inclusion_weight'),
            ('(((htxs_stage1p1cat==100||htxs_stage1p1cat==102||htxs_stage1p1cat==103)*crossSectionPerEventWeight*numberGeneratedEventsWeight+'
             '(htxs_stage1p1cat==101)*2.09e-8+'
             '(htxs_stage1p1cat==104||htxs_stage1p1cat==105)*4.28e-8+'
             '(htxs_stage1p1cat==106)*1.39e-8+'
             '(htxs_stage1p1cat>=107&&htxs_stage1p1cat<=109)*4.90e-8+'
             '(htxs_stage1p1cat>=110&&htxs_stage1p1cat<=113)*9.69e-9'
             ')*(abs(crossSectionPerEventWeight - 0.00538017) > 1e-5) + numberGeneratedEventsWeight*crossSectionPerEventWeight*(abs(crossSectionPerEventWeight - 0.00538017) < 1e-5))', 'ggh_stitching_weight')
            ],
        cuts = [
            ('(htxs_stage1p1cat>=100)&&(htxs_stage1p1cat<=113)', 'htxs_cut')
            ]
        )

qqh = Selection(name = 'qqh',
        weights = [
            ('((htxs_stage1p1cat>=200&&htxs_stage1p1cat<=202)||abs(crossSectionPerEventWeight-0.04774)<0.001||abs(crossSectionPerEventWeight-0.052685)<0.001||abs(crossSectionPerEventWeight-0.03342)<0.001)*crossSectionPerEventWeight*numberGeneratedEventsWeight+(abs(crossSectionPerEventWeight-0.04774)>=0.001&&abs(crossSectionPerEventWeight-0.052685)>=0.001&&abs(crossSectionPerEventWeight-0.03342)>=0.001)*('
             '(htxs_stage1p1cat>=203&&htxs_stage1p1cat<=205)*9.41e-9+'
             '(htxs_stage1p1cat==206)*8.52e-9+'
             '(htxs_stage1p1cat>=207&&htxs_stage1p1cat<=210)*1.79e-8'
             ')', 'qqh_stitching_weight')
            ],
        cuts = [
            ('(htxs_stage1p1cat>=200)&&(htxs_stage1p1cat<=210)', 'htxs_cut')
            ]
        )

# Variations

same_sign = ReplaceCut('same_sign', 'os', Cut('q_1*q_2>0', 'ss'))

ggh_wg1 = []
for unc in ['THU_ggH_Mig01', 'THU_ggH_Mig12', 'THU_ggH_Mu', 'THU_ggH_PT120', 'THU_ggH_PT60',
            'THU_ggH_Res', 'THU_ggH_VBF2j', 'THU_ggH_VBF3j', 'THU_ggH_qmtop']:
    ggh_wg1.append(AddWeight(unc + 'Up', Weight('({})'.format(unc), '{}_wg1'.format(unc))))
    ggh_wg1.append(AddWeight(unc + 'Down', Weight('(1.0/{})'.format(unc), '{}_wg1'.format(unc))))

qqh_wg1 = []
for unc in ['THU_qqH_25', 'THU_qqH_JET01', 'THU_qqH_Mjj1000', 'THU_qqH_Mjj120', 'THU_qqH_Mjj1500',
            'THU_qqH_Mjj350', 'THU_qqH_Mjj60', 'THU_qqH_Mjj700', 'THU_qqH_PTH200', 'THU_qqH_TOT']:
    qqh_wg1.append(AddWeight(unc + 'Up', Weight('({})'.format(unc), '{}_wg1'.format(unc))))
    qqh_wg1.append(AddWeight(unc + 'Down', Weight('(1.0/{})'.format(unc), '{}_wg1'.format(unc))))

jet_es = []
for name in ['Absolute', 'BBEC1', 'EC2', 'HF']:
    jet_es += [ChangeDataset('CMS_scale_j_{}_2018Up'.format(name), 'jecUnc{}YearUp'.format(name)),
               ChangeDataset('CMS_scale_j_{}_2018Down'.format(name), 'jecUnc{}YearDown'.format(name)),
               ChangeDataset('CMS_scale_j_{}Up'.format(name), 'jecUnc{}Up'.format(name)),
               ChangeDataset('CMS_scale_j_{}Down'.format(name), 'jecUnc{}Down'.format(name))]
jet_es += [
        ChangeDataset('CMS_scale_j_RelativeBalUp', 'jecUncRelativeBalUp'),
        ChangeDataset('CMS_scale_j_RelativeBalDown', 'jecUncRelativeBalDown'),
        ChangeDataset('CMS_scale_j_RelativeSample_2018Up', 'jecUncRelativeSampleYearUp'),
        ChangeDataset('CMS_scale_j_RelativeSample_2018Down', 'jecUncRelativeSampleYearDown'),
        ChangeDataset('CMS_scale_j_FlavorQCDUp', 'jecUncFlavorQCDUp'),
        ChangeDataset('CMS_scale_j_FlavorQCDDown', 'jecUncFlavorQCDDown'),
        ChangeDataset('CMS_res_j_2018Up', 'jerUncUp'),
        ChangeDataset('CMS_res_j_2018Down', 'jerUncDown'),
        ]

tau_es = [
        ChangeDataset('CMS_scale_t_3prong_2018Up', 'tauEsThreeProngUp'),
        ChangeDataset('CMS_scale_t_3prong_2018Down', 'tauEsThreeProngDown'),
        ChangeDataset('CMS_scale_t_3prong1pizero_2018Up', 'tauEsThreeProngOnePiZeroUp'),
        ChangeDataset('CMS_scale_t_3prong1pizero_2018Down', 'tauEsThreeProngOnePiZeroDown'),
        ChangeDataset('CMS_scale_t_1prong_2018Up', 'tauEsOneProngUp'),
        ChangeDataset('CMS_scale_t_1prong_2018Down', 'tauEsOneProngDown'),
        ChangeDataset('CMS_scale_t_1prong1pizero_2018Up', 'tauEsOneProngOnePiZeroUp'),
        ChangeDataset('CMS_scale_t_1prong1pizero_2018Down', 'tauEsOneProngOnePiZeroDown')
        ]

# Binnings

binning = {
    'npv': [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,89],
    'm_vis': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'pt_1': [0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'pt_2': [0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'eta_1': [-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
    'eta_2': [-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
    'jpt_1': [0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'jpt_2': [0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'jeta_1': [-4.8, -4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8],
    'jeta_2': [-4.8, -4.6, -4.4, -4.2, -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8],
    'njets': [0,1,2,3,4,5],
    'nbtag': [0,1,2,3,4,5],
    'mt_1': [0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45,47.5,50,55],
    'mt_2': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'ptvis': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'pt_tt': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'mjj': [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300],
    'jdeta': [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0],
    'dijetpt': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'met': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'm_sv_puppi': [0,7.5,15,22.5,30,37.5,45,52.5,60,67.5,75,82.5,90,97.5,105,112.5,120,127.5,135,142.5,150,157.5,165,172.5,180,187.5,195,202.5,210,217.5,225],
    'pt_sv_puppi': [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160],
    'eta_sv_puppi': [-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    }

control_variables = list(binning.keys())

# Variables used for ML training

ml_variables = ['njets', 'nbtag', 'ptvis', 'pt_tt', 'mjj', 'jdeta', 'dijetpt', 'met', 'm_vis', 'm_sv_puppi']
ml_weight = 'training_weight'
ml_classes = ['ggh', 'qqh', 'ztt', 'zl', 'w', 'tt', 'vv']

# Analysis categories

analysis_categories = {'nll_cat': Selection(name = 'nll_cat')}
analysis_binning = np.linspace(0, 1, 9)
analysis_variable = 'ml_score'
