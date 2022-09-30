import class_ggr
from class_ggr import GGR_GRU4Rec



if __name__ == '__main__':
    website_path = 'csv_file/website.csv'
    ggru = GGR_GRU4Rec(path=website_path, pretrain=True)
    print(ggru.show_prediction(20))
    # ggru.to_csv()
    # print(ggru.pred_to_list(666))
    # print(ggru.to_dict()[666]['prediction'])


