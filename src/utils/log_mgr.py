from prettytable import PrettyTable


class LogMgr:
    def __init__(self, logger):
        self.name2val = dict()
        self.name2cnt = dict()
        self.logger = logger

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        if key not in self.name2val:
            self.name2val[key] = val
            self.name2cnt[key] = 1
        else:
            self.name2val[key] = (self.name2val[key] * self.name2cnt[key] + val) / (self.name2cnt[key] + 1)
            self.name2cnt[key] += 1

    def dumpkvs(self):
        td = PrettyTable(['Key', 'Value', 'Count'])
        keys = ['lr']
        keys += sorted([i for i in self.name2val.keys() if i.startswith('loss')])
        keys += [i for i in self.name2val.keys() if not i.startswith('loss') and i != 'lr']
        for k in keys:
            v = self.name2val[k]
            v = v if type(v) == str else f'{v:.6f}' if type(v) != int else v
            td.add_row([k, v] + ([self.name2cnt[k], ] if k in self.name2cnt else ['-', ]))

        td.align['Key'] = "l"
        td.align['Value'] = 'r'
        td.align['Count'] = 'r'
        self.logger(td)

        self.name2val = dict()
        self.name2cnt = dict()
