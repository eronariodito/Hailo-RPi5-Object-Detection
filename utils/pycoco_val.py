import contextlib
from colorama import Fore, Style
import io

def validate_pycoco(self, preds_file):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(preds_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    print_pycoco(self, coco_eval.stats)
    return

def print_pycoco(self, stats):
    self.logger.info(f"\n{Fore.CYAN}📊 {'Average Precision (AP)'}{Style.RESET_ALL}\n")
    
    # AP values
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.50:0.95  | area=all    | maxDets=100 ]:{Style.RESET_ALL} {stats[0]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.50       | area=all    | maxDets=100 ]:{Style.RESET_ALL} {stats[1]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.75       | area=all    | maxDets=100 ]:{Style.RESET_ALL} {stats[2]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.50:0.95  | area=small  | maxDets=100 ]:{Style.RESET_ALL} {stats[3]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.50:0.95  | area=medium | maxDets=100 ]:{Style.RESET_ALL} {stats[4]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AP {Style.RESET_ALL}@[ IoU=0.50:0.95  | area=large  | maxDets=100 ]:{Style.RESET_ALL} {stats[5]:.3f}")
    
    # AR values
    self.logger.info(f"\n{Fore.CYAN}📈 {'Average Recall (AR)'}{Style.RESET_ALL}\n")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=all     | maxDets=  1 ]:{Style.RESET_ALL} {stats[6]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=all     | maxDets= 10 ]:{Style.RESET_ALL} {stats[7]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=all     | maxDets=100 ]:{Style.RESET_ALL} {stats[8]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=small   | maxDets=100 ]:{Style.RESET_ALL} {stats[9]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=medium  | maxDets=100 ]:{Style.RESET_ALL} {stats[10]:.3f}")
    self.logger.info(f"{Fore.YELLOW}AR {Style.RESET_ALL}@[ IoU=0.50:0.95 | area=large   | maxDets=100 ]:{Style.RESET_ALL} {stats[11]:.3f}\n")