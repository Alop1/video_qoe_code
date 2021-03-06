from collections import Counter
from matplotlib import pyplot as plt


def create_feature_bar_chart(DB, type, save=False ):
    if "fsp" in type:
        feature_sumup = Counter(DB.fps)
    elif "resolution" in type:
        feature_sumup = Counter(DB.Resolution)
    elif "color_encoding" in type:
        feature_sumup = Counter(DB.Color_encoding)
    else:
        print "usage create_feature_bar_chart : fsp, resolution, color_encoding"
        return 0
    feature, counters = zip(*feature_sumup.items())
    counters = [int(counter) for counter in counters ]
    print "feature {} counter {} ".format(feature, counters)
    xs = [i for i, _ in enumerate(feature_sumup)]
    plt.bar(xs, counters)
    plt.ylabel("liczba filmow")
    plt.title(type)
    plt.xticks(xs, feature)
    plt.xlabel(type)
    if save:
        plt.savefig("reports/{}.png".format(type))
    else:
        plt.show()
    plt.clf()

def create_quantitative_to_mos_chart(datafarme, feature, save=False):
    plt.scatter(datafarme[feature], datafarme["Subject_score"])
    plt.title("feature to subject score chart")
    plt.ylabel("Subject score")
    if "Aggregate_vmaf" in feature:
        feature = "VMAF"
    plt.xlabel(feature)
    if save:
        plt.savefig("reports/{}.png".format(feature))
    else:
        plt.show()
    plt.clf()

def create_nominal_value_to_mos_chart(DB, type, save=False):
    column_name = ''
    if "fps" in type:
        column_name = 'fps'
        print DB["fps"]
        feature_sumup = Counter(DB.fps)
    elif "resolution" in type:
        column_name = "Resolution"
        feature_sumup = Counter(DB.Resolution)
    elif "color_encoding" in type:
        column_name = "Color_encoding"
        feature_sumup = Counter(DB.Color_encoding)
    else:
        print "usage create_feature_bar_chart : fsp, resolution, color_encoding"
        return 0
    features, counters = zip(*feature_sumup.items())
    print features
    mean_mos_value_mapper = {}
    feature_names = []
    for feature in features:
        mos_values_per_feature = list(DB.loc[DB[column_name] == feature]["Subject_score"].values)
        print mos_values_per_feature
        mean_mos = sum(mos_values_per_feature)/len(mos_values_per_feature)
        mean_mos_value_mapper[feature] = str(mean_mos)
        # print "mean_mos_value_mapper: ", mean_mos_value_mapper
        feature_names.append(feature)
        print "feature: ", feature

    # feature_names = ["\n".join(wrap(name,12)) for name in feature_names]
    counters = [int(counter) for counter in counters]
    print "feature {} counter {} ".format(features, counters)
    print
    xs = [i for i, _ in enumerate(feature_sumup)]
    plt.bar(xs, counters)
    plt.ylabel("video quantity")
    plt.title(type)
    plt.xticks(xs, features)
    plt.xlabel(type)
    for i, ele in enumerate(mean_mos_value_mapper.values()):
        ele = "mean mos: " + ele
        plt.text(x=xs[i]-0.5, y=counters[i]+3, s=ele, size=8)
    if save:
        plt.savefig("reports/{}.png".format(type))
    else:
        plt.show()
    plt.clf()

def create_correlation_chart(merged_DB):
    import numpy as np
    from data_analysis.models import map_to_polish
    data_for_correlation = merged_DB[["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM",
                                      "blockiness", "spatialactivity", "pillarbox",
                                      "blockloss", "blur", "temporalact", "blockout",
                                      "exposure", "contrast", "brightness", 'duration']]
    data_for_correlation = merged_DB[["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM",
                                      "blockiness", "pillarbox",
                                      "blockloss", "blur", "temporalact", "blockout",
                                      "exposure", "contrast", "spatialactivity", "brightness", ]]
    # data_for_correlation = merged_DB[["PSNR", "Aggregate_vmaf", "SSIM", "MS_SSIM"]]
    labels_names = []
    mapping_dir = map_to_polish()
    for column_name in data_for_correlation.columns:
        if column_name in mapping_dir:
            labels_names.append(mapping_dir[column_name])
        else:
            labels_names.append(column_name)

    DB_copy = data_for_correlation.copy(deep=True)
    corr = DB_copy.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data_for_correlation.columns), 1)
    x_ticks = ticks.copy()
    x_ticks[-2] = 12.7
    ax.set_xticks(x_ticks)
    # plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    # ax.set_xticklabels(labels_names, )
    ax.set_xticklabels(labels_names, rotation=40)
    ax.set_yticklabels(labels_names, rotation=45)
    plt.show()

