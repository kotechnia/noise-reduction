import pandas as pd
import json

def decibel_to_level(decibel):
    if decibel < 86:
        return "76dB~85dB"
    elif decibel < 96:
        return "86dB~95dB"
    else:
        return "96dB~"

def main(args):

    results_path = args.results_path

    # 결과 파일을 연결시킵니다. 1개 이상 복수개의 결과 합산
    df_results=[]
    for result_path in results_path:
        df_results.append(pd.read_csv(result_path))
    df_results = pd.concat(df_results)

    # key가 중복이 있을 경우 첫번째 키만 사용합니다.
    df_results = df_results.drop_duplicates(subset=['key']).reset_index(drop=True)

    
    # 데시벨 정보를 읽어들입니다.
    for i in range(len(df_results)):
        json_path = df_results.loc[i, 'script_path']

        # script_path의 json을 파싱합니다.
        with open(json_path) as f:
            data = json.load(f)
            decibel = data['noiseInfo']['bgnoisespl']
        df_results.loc[i, 'decibel'] = float(decibel)

    # 데시벨을 76dB~85dB, 86dB~95dB, 96dB~ 로 그룹하기 위하여 변환합니다.
    df_results['level'] = df_results['decibel'].map(decibel_to_level)

    # 그룹화된 결과들을 평균합니다.
    df_report = df_results.groupby(by=['level']).mean()[['estoi', 'f1_error_rate']]


    # 모든 총합산 ALL에 대한 계산
    total_estoi = df_results['estoi'].mean()
    total_f1_error_rate = df_results['f1_error_rate'].mean()
    total_score = pd.DataFrame([{'level':'all', 'estoi':total_estoi, 'f1_error_rate':total_f1_error_rate}])
    total_score = total_score.set_index('level')
    df_report = pd.concat([df_report, total_score])


    # 해당 결과를 출력합니다.
    print(df_report)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', metavar='N', type=str, nargs='+')
    args = parser.parse_args()

    # command
    # python report_denoise.py --results_path results.csv

    # 
    # python report_denoise.py --results_path results_0_632.csv results_632_1265.csv results_1265_1898.csv results_1898_2531.csv
    #
    #                            estoi  f1_error_rate
    #              level                             
    #              76dB~85dB  0.850338       0.242371
    #              86dB~95dB  0.850333       0.190160
    #              96dB~      0.863482       0.146552
    #              all        0.854023       0.191810
    
    main(args)
