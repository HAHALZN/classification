from tbparse import SummaryReader
#
# if __name__ == '__main__':
#     log_dir = "../checkpoints/train_target/RDR_TARGET_EFFICIENTNETB5"
#
#     print(SummaryReader(log_dir, pivot=True).scalars.to_csv('../checkpoints/summary.csv'))

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_acc = EventAccumulator(r'../checkpoints/train_target/RDR_TARGET_EFFICIENTNETB5/20220506150233_efficientnet_b5_tgt_A_src_DIMM2_bs_32_lr_0.005_beta_0.3_gamma_0.01_par_0.3_size_full/events.out.tfevents.1651820553.zyht04-System-Product-Name.12463.0')
event_acc.Reload()
print(event_acc.Tags())
for e in event_acc.Scalars('Accuracy/top-1'):
    print(e.step, e.value)