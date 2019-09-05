import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.3
from matplotlib.patches import Rectangle

np.random.seed(215)
# COLORS = ["#247BA0",
#           "#70C1B3",
#           "#B2DBBF",
#           "#F3FFBD",
#           "#FF1654"]
# COLORS = ["#abd8ea",
#           "#37aff7",
#           "#2f97ef",
#           "#178fd7",
#           "#0077c0"]
COLORS = ["#deecfb",
          "#bedaf7",
          "#7ab3ef",
          "#368ce7",
          "#1666ba"]
# COLORS = ["#e6f0e1",
#           "#d7ecd1",
#           "#bbdfba",
#           "#a4cca5",
#           "#8bbf8c"]

PAD_COLOR = "#F0F0F0"
PAD_ALPHA = 1.0
PAD_LINEWIDTH = 1.5
ALPHA = 1.0
RECT_LINEIWDTH = 0.5
EDGECOLOR = 'k'
BAR_HEIGHT = 5
bucket_ranges = [(5, 10),
                 (10, 15),
                 (15, 20),
                 (20, 25),
                 (25, 30)]
bucket_label = ["[5, 10)",
                "[10, 15)",
                "[15, 20)",
                "[20, 25)",
                "[25, 30)"]
bucket_nums = [48, 32, 16, 8, 8]

MAX_LEN = 30
BASE_RATIO = 20.0 / float(sum(bucket_nums))
PAD_LEN = 0.005
BATCH_SIZE = 8


def plot_seq(seq_info, x_begin, y_begin, x_end, y_end, save_path=None, title=None):
    fig, ax = plt.subplots(figsize=(len(seq_info) * BASE_RATIO, 5))
    ax.set_axis_off()
    x_len = (x_end - x_begin) / float(len(seq_info))
    y_len = float(y_end - y_begin)
    legend_objs = [None for _ in range(len(bucket_label))]
    # Draw Rectangles
    for i, (seq_len, bucket_id) in enumerate(seq_info):
        rect = Rectangle((x_begin + x_len * i, y_begin), x_len,
                         y_len * float(seq_len) / MAX_LEN,
                         facecolor=COLORS[bucket_id],
                         linewidth=RECT_LINEIWDTH,
                         edgecolor=EDGECOLOR,
                         alpha=ALPHA)
        legend_objs[bucket_id] = rect
        ax.add_patch(rect)

    fig.legend(legend_objs, bucket_label, loc="upper center", ncol=len(legend_objs), borderaxespad=0.05, fontsize=12)
    title = r'Data Samples' if title is None else title
    ax.text(0.5, - 0.04, title, horizontalalignment='center',
            verticalalignment='center', fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def plot_bucket_seq(seq_info, x_begin, y_begin, x_end, y_end, bucket_sizes, save_path=None, title=None, sort_length=False):
    fig, ax = plt.subplots(figsize=(len(seq_info) * BASE_RATIO, 5))
    ax.set_axis_off()
    bucket_num = len(bucket_sizes)
    x_len = (x_end - x_begin - PAD_LEN * (bucket_num - 1)) / float(len(seq_info))
    y_len = float(y_end - y_begin)
    legend_objs = [None for _ in range(len(bucket_label))]
    # Draw Rectangles
    sample_id = 0
    print(bucket_sizes)
    for bucket_id, size in enumerate(bucket_sizes):
        bucket_seq_info = seq_info[sample_id:(sample_id + size)]
        if sort_length:
            bucket_seq_info = sorted(bucket_seq_info, key=lambda ele: ele[0], reverse=True)
        for i in range(size):
            seq_len, color_id = bucket_seq_info[i]
            rect = Rectangle((x_begin + x_len * sample_id + bucket_id * PAD_LEN, y_begin), x_len,
                             y_len * float(seq_len) / MAX_LEN,
                             facecolor=COLORS[color_id],
                             linewidth=RECT_LINEIWDTH,
                             edgecolor=EDGECOLOR,
                             alpha=ALPHA)
            sample_id += 1
            legend_objs[color_id] = rect
            ax.add_patch(rect)
        ax.annotate(r'$Bucket_{%d}$' %bucket_id,
                    xy=(x_begin + x_len * (sample_id - size / 2.0) + bucket_id * PAD_LEN, y_begin - 0.02),
                    xytext=(x_begin + x_len * (sample_id - size / 2.0) + bucket_id * PAD_LEN, y_begin - 0.1),
                    xycoords='axes fraction',
                    fontsize=12, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-[, widthB=%g, lengthB=0.5' % (size * x_len * 55), lw=1.0))
    fig.legend(legend_objs, bucket_label, loc="upper center", ncol=len(legend_objs), borderaxespad=0.05, fontsize=12)
    title = r'Bucket Data Samples' if title is None else title
    ax.text(0.5, - 0.04, title, horizontalalignment='center',
            verticalalignment='center', fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)


def add_padded_batches(ax, all_batch_inds, seq_info, x_begin, y_begin, x_end, y_end):
    batch_num = len(all_batch_inds)
    cnt = 0
    padding_val = 0.0
    legend_objs = [None for _ in range(len(bucket_label) + 1)]
    x_len = (x_end - x_begin - PAD_LEN * (batch_num - 1)) / float(len(seq_info))
    y_len = float(y_end - y_begin)
    for bid, batch_inds in enumerate(all_batch_inds):
        max_seq_len = max(seq_info[j][0] for j in batch_inds)
        rect = Rectangle((x_begin + x_len * cnt + bid * PAD_LEN, y_begin),
                         len(batch_inds) * x_len,
                         y_len * float(max_seq_len) / MAX_LEN,
                         facecolor=PAD_COLOR,
                         linewidth=PAD_LINEWIDTH,
                         linestyle='-',
                         edgecolor=EDGECOLOR,
                         hatch='/',
                         alpha=PAD_ALPHA,
                         zorder=1)
        ax.add_patch(rect)
        border_rect = Rectangle((x_begin + x_len * cnt + bid * PAD_LEN, y_begin),
                         len(batch_inds) * x_len,
                         y_len * float(max_seq_len) / MAX_LEN,
                         linewidth=PAD_LINEWIDTH,
                         linestyle='-',
                         edgecolor=EDGECOLOR,
                         fill=False,
                         zorder=20)
        ax.add_patch(border_rect)
        ax.text(x_begin + x_len * cnt + bid * PAD_LEN + len(batch_inds) * x_len / 2.0,
                y_begin - 0.03,
                r'$B_{%d}$' % bid, horizontalalignment='center', verticalalignment='center',
                fontsize=12)
        legend_objs[0] = rect

        for ind in batch_inds:
            seq_len, color_id = seq_info[ind]
            rect = Rectangle((x_begin + x_len * cnt + bid * PAD_LEN, y_begin), x_len,
                             y_len * float(seq_len) / MAX_LEN,
                             facecolor=COLORS[color_id],
                             linewidth=RECT_LINEIWDTH,
                             edgecolor=EDGECOLOR,
                             alpha=ALPHA,
                             label=bucket_label[bucket_id],
                             zorder=10)
            cnt += 1
            legend_objs[color_id + 1] = rect
            ax.add_patch(rect)
            padding_val += max_seq_len - seq_len
    avg_padding = padding_val / float(len(seq_info))
    return avg_padding, legend_objs

def plot_batches(seq_info, all_batch_inds, x_begin, y_begin, x_end, y_end, save_path=None,
                 title="Bucketing Strategy. "):
    fig, ax = plt.subplots(figsize=(len(seq_info) * BASE_RATIO, 5))
    ax.set_axis_off()
    avg_pad, legend_objs = add_padded_batches(ax, all_batch_inds, seq_info, x_begin, y_begin, x_end, y_end)
    fig.legend(legend_objs, ['padding'] + bucket_label, loc="upper center", ncol=len(legend_objs),
               borderaxespad=0.04, fontsize=12)
    ax.text(0.5, - 0.04,
            r'%sAvg Pad = %.1f' % (title, avg_pad),
            horizontalalignment='center', verticalalignment='center', fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print('%savg padding=%g' %(title, avg_pad))
    return avg_pad


def get_no_bucket_inds(seq_info):
    batch_inds = []
    for begin in range(0, len(seq_info), BATCH_SIZE):
        end = min(begin + BATCH_SIZE, len(seq_info))
        batch_inds.append(list(range(begin, end)))
    return batch_inds


def get_sorted_bucket_inds(seq_info, mult=4):
    batch_inds = []
    for bucket_begin in range(0, len(seq_info), BATCH_SIZE * mult):
        bucket_end = min(bucket_begin + BATCH_SIZE * mult, len(seq_info))
        bucket_sample_ids = sorted(range(bucket_begin, bucket_end),
                                   key=lambda ele: seq_info[ele][0],
                                   reverse=True)
        for begin in range(0, bucket_end - bucket_begin, BATCH_SIZE):
            end = min(begin + BATCH_SIZE, bucket_end - bucket_begin)
            batch_inds.append(bucket_sample_ids[begin:end])
    return batch_inds


def get_fixed_bucket_inds(seq_info, ratio=0.0):
    bucket_sample_ids = [[] for _ in range(5)]
    batch_inds = []
    for i, (seq_len, color_id) in enumerate(seq_info):
        bucket_sample_ids[color_id].append(i)
    bucket_seq_len = [ele[1] - 1 for ele in bucket_ranges]

    bucket_batch_sizes = [max(int(BATCH_SIZE * ratio * max(bucket_seq_len) / float(ele_len)), BATCH_SIZE)
                         for ele_len in bucket_seq_len]
    bucket_sample_ids, bucket_batch_sizes = bucket_sample_ids[::-1], bucket_batch_sizes[::-1]
    for i, (sample_ids, batch_size) in enumerate(zip(bucket_sample_ids, bucket_batch_sizes)):
        for begin in range(0, len(sample_ids), batch_size):
            end = min(begin + batch_size, len(sample_ids))
            batch_inds.append(sample_ids[begin:end])
    return bucket_sample_ids, batch_inds


seq_info = []
for bucket_id, (brange, bnum) in enumerate(zip(bucket_ranges, bucket_nums)):
    for _ in range(bnum):
        seq_info.append((np.random.randint(brange[0], brange[1]), bucket_id))
np.random.shuffle(seq_info)
plot_seq(seq_info, 0.0, 0.0, 0.99, 0.97, save_path="data_samples.png")
batch_inds = get_no_bucket_inds(seq_info)
plot_batches(seq_info, batch_inds, 0.0, 0.05, 0.99, 0.97,
             save_path="no_bucket_strategy.png", title="No Bucketing Strategy. ")

sorted_bucket_size = [BATCH_SIZE * 4 for _ in range(0, len(seq_info), BATCH_SIZE * 4)]
sorted_bucket_size[-1] -= BATCH_SIZE * 4 * len(sorted_bucket_size) - len(seq_info)
plot_bucket_seq(seq_info, 0.0, 0.1, 0.99, 0.97, bucket_sizes=sorted_bucket_size, save_path="sorted_bucket_data_samples.png", title=r"Data Samples. Bucket Size = %d" %(BATCH_SIZE * 4))
plot_bucket_seq(seq_info, 0.0, 0.1, 0.99, 0.97, bucket_sizes=sorted_bucket_size, save_path="sorted_bucket_data_samples_after_sort.png", title=r"Sorted Data Samples. Bucket Size = %d" %(BATCH_SIZE * 4), sort_length=True)
# plot_seq_batch_size_padded(seq_info, 0.0, 0.05, 0.99, 0.97, batch_size=BATCH_SIZE, save_path="no_bucket_strategy.png")

batch_inds = get_sorted_bucket_inds(seq_info, mult=4)
plot_batches(seq_info, batch_inds, 0.0, 0.05, 0.99, 0.97,
             save_path="sorted_bucket_strategy.png", title=r"Sorted Bucketing Strategy. Bucket Size = %d, " % (BATCH_SIZE * 4))


bucket_sample_ids, batch_inds = get_fixed_bucket_inds(seq_info, 0.0)
plot_batches(seq_info, batch_inds, 0.0, 0.05, 0.99, 0.97,
             save_path="fixed_bucket_strategy_ratio0.0.png", title="Fixed Bucketing Strategy. Ratio = 0.0, ")

plot_bucket_seq([seq_info[i] for i in sum(bucket_sample_ids, [])], 0.0, 0.1, 0.99, 0.97,
                bucket_sizes=[len(ele) for ele in bucket_sample_ids],
                save_path="fixed_bucket_data_samples.png", title=r"Reorganized Data Samples")

bucket_sample_ids, batch_inds = get_fixed_bucket_inds(seq_info, 0.7)
plot_batches(seq_info, batch_inds, 0.0, 0.05, 0.99, 0.97,
             save_path="fixed_bucket_strategy_ratio0.7.png", title="Fixed Bucketing Strategy. Ratio = 0.7, ")


plt.show()
