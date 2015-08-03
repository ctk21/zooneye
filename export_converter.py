
import argparse
import math
import numpy as np
import os
import pandas as pd
import sys
import yaml

## valid workflow id and minimum versions supported
VALID_WORKFLOWS = {
    162: 14.23,        ## 'NerveMarking1'
    }

TASK_KEY_DISK_BOX       = (162, 'T1')
TASK_KEY_MARK_FOVEA     = (162, 'T3') 
TASK_KEY_CUP_BOUNDARY   = (162, 'T4')
TASK_KEY_DISK_BOUNDARY  = (162, 'T5')
TASK_KEY_MARK_NOTCH_HAEMORRHAGE = (162, 'T6') 

CSV_KEY_ORDER = ['created_at', 'user_name', 'expert', 'subject_id', 'subject_filename']

def df_xcols(df, cols):
    xs = np.concatenate((cols, df.columns.values))
    _, idx = np.unique(xs, return_index=True)
    return df[xs[np.sort(idx)]]

def row_ukey(row):
    return {'user_name': row['user_name'], 'created_at': row['created_at'], 'expert': nan2value(row['expert'], 0)}

def row_skey(row):
    ## NB: assumes only one subject per classification
    sub_dat = parse_field(row['subject_data'])
    sub_key = next(iter(sub_dat.keys()))
    return {'subject_id': sub_key, 'subject_filename': sub_dat[sub_key]['filename']}

def nan2value(x, v):
    return x if not math.isnan(x) else v

def parse_field(field):
    return yaml.safe_load(field) if type(field) is str else field

def parse_point_array(point_list):
    return pd.DataFrame(point_list).values

def is_task_key(task_key, workflow_id, annotation_row, skip_empty_value=True):
    if workflow_id != task_key[0]:
        return False
    if ('task' in annotation_row) and ('task_label' in annotation_row):
        return annotation_row['task'] == task_key[1] and \
               ((not skip_empty_value) or len(annotation_row['value']) > 0)
    else:
        print('No task, task_label with value in annotation: %s'%str(annotation_row))
        return False

def push_keys_and_dict_onto_list(ukey, skey, rdict, the_list):
    rdict.update(ukey)
    rdict.update(skey)
    the_list.append(rdict)
    return the_list

class BaseAccumulator(object):
    def setup(self, df): pass        
    def handle_row(self, rkey, skey, row): pass
    def finish(self, df): pass
    
    
class AccumulateWorkflows(BaseAccumulator):
    def finish(self, df):
        print('Workflow stats:')
        print(df.groupby(['workflow_id', 'workflow_name', 'workflow_version'])['user_name'].count())
        

class AccumulateTasks(BaseAccumulator):
    def setup(self, df):
        self.task_dict = {}
        
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if ('task' in x) and ('task_label' in x):
                k = (workflow_id, x['task'], x['task_label'])
                if k not in self.task_dict: self.task_dict[k] = 0
                self.task_dict[k] += 1
            else:
                print('No task, task_label in annotation of row: %s|%s item dump: %s'%(rkey, skey, str(x)))

    def finish(self, df):
        print('Task label stats:')
        for k, v in iter(self.task_dict.items()):
            print("%s: %i"%(k,v))


class AccumulateOpticNerveBox(BaseAccumulator):
    def __init__(self, out_file=None):
        self.out_file = out_file
    
    def setup(self, df):
        self.row_list = []
        
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_DISK_BOX, workflow_id, x):
                dat = x['value'][0]
                rdict = {'height': dat['height'],
                         'width': dat['width'],
                         'x': dat['x'],
                         'y': dat['y'],}
                push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)

    def finish(self, df):        
        stat_df = df_xcols(pd.DataFrame(self.row_list), CSV_KEY_ORDER)
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['x'].agg({'n': len}),
                       grouped[['x', 'y', 'width', 'height']].agg(np.mean))
        print('Optic Nerve Box statistics: ')
        print(tmp)
        print('Count Optic Nerve Box rows: %i'%len(stat_df))
        print('Mean Optic Nerve Box height/width: %.4f'%np.mean(stat_df['height'].values/stat_df['width'].values))

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
        
            
class AccumulateFoveaMarks(BaseAccumulator):
    def __init__(self, out_file=None):
        self.out_file = out_file
    
    def setup(self, df):
        self.row_list = []
     
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_MARK_FOVEA, workflow_id, x):
                dat = x['value'][0]
                rdict = {'x': dat['x'],
                         'y': dat['y'],}
                push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)

    def finish(self, df):        
        stat_df = df_xcols(pd.DataFrame(self.row_list), CSV_KEY_ORDER)
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['x'].agg({'n': len}),
                       grouped[['x', 'y']].agg(np.mean))
        print('Fovea mark statistics: ')
        print(tmp)

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
                

class AccumulateCupDiskBoundaryBox(BaseAccumulator):
    def __init__(self, out_file=None):
        self.out_file = out_file
        
    def setup(self, df):
        self.row_list = []

    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        valid_disk = valid_cup = False
        for x in annotations:
            if is_task_key(TASK_KEY_DISK_BOUNDARY, workflow_id, x):
                dat = x['value']
                if not 'points' in dat[0]:
                    print('WARNING: skipping disk boundary as no points field: %s'%str(x))
                else:
                    dat = dat[0]
                    assert(dat['tool'] == 0)
                    points = parse_point_array(dat['points'])

                    disk_min = np.min(points, axis=0)
                    disk_max = np.max(points, axis=0)                    
                    valid_disk = True

            if is_task_key(TASK_KEY_CUP_BOUNDARY, workflow_id, x):
                dat = x['value']
                if not 'points' in dat[0]:
                    print('WARNING: skipping cup boundary as no points field: %s'%str(x))
                else:                    
                    dat = dat[0]
                    assert(dat['tool'] == 0)
                    points = parse_point_array(dat['points'])

                    cup_min = np.min(points, axis=0)
                    cup_max = np.max(points, axis=0)
                    valid_cup = True
        
        if valid_cup and valid_disk:
            rdict = {
                'disk_x': disk_min[0],
                'disk_y': disk_min[1],
                'disk_width': disk_max[0] - disk_min[0],
                'disk_height': disk_max[1] - disk_min[1],
                'cup_x': cup_min[0],
                'cup_y': cup_min[1],
                'cup_width': cup_max[0] - cup_min[0],
                'cup_height': cup_max[1] - cup_min[1],            
            }
            rdict['vertical_cdr'] = rdict['cup_height']/rdict['disk_height']
            rdict['horizontal_cdr'] = rdict['cup_width']/rdict['disk_width']
            push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)
        
    def finish(self, df):
        stat_df = df_xcols(pd.DataFrame(self.row_list), CSV_KEY_ORDER)
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['disk_x'].agg({'n': len}),
                       grouped[['vertical_cdr', 'horizontal_cdr']].agg(np.mean))

        print('CupDiskBoundaryBox statistics: ')
        print(tmp)

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
        
   
class AccumulateNotchHaemorrhageMarks(BaseAccumulator):
    def __init__(self, out_file=None):
        self.out_file = out_file
    
    def setup(self, df):
        self.row_list = []
     
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_MARK_NOTCH_HAEMORRHAGE, workflow_id, x, skip_empty_value=False):
                for mark in x['value']:
                    rdict = { 'mark_id': mark['tool'],
                              'mark_label': mark['tool_label'],
                              'x': mark['x'],
                              'y': mark['y'], }
                    push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)    

                if len(x['value']) == 0:
                    rdict = { 'mark_id': -1, 
                              'mark_label': 'No_Notch_Or_Haemorrhage',
                              'x': -1,
                              'y': -1, }
                    push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)    
                    
    def finish(self, df):        
        stat_df = df_xcols(pd.DataFrame(self.row_list), CSV_KEY_ORDER)
        grouped = stat_df.groupby(['subject_id', 'mark_id', 'mark_label'], as_index=False)
        tmp = grouped['x'].agg({'n': len})
        print('Notch/Haemorrhage mark statistics: ')
        print(tmp)

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)

            
class PrintSubjectInfo(BaseAccumulator):
    def handle_row(self, rkey, skey, row):
        subject = parse_field(row['subject_data'])
        print(yaml.dump(subject))
                           
class PrintRowsForTaskKey(BaseAccumulator):
    def __init__(self, tkey):
        self.tkey = tkey

    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(self.tkey, workflow_id, x):
                print('rkey: "%s"'%str(rkey))
                print(yaml.dump(x))
                
                
def main(args):
    for fnme in args.file:
        if args.verbose: print('Processing: '+fnme)
        
        df = pd.read_csv(fnme)   
        if args.verbose: df.info()

        ## filter to valid workflows
        df = df.loc[df['workflow_id'].isin(VALID_WORKFLOWS.keys())]
        min_id = df['workflow_id'].replace(VALID_WORKFLOWS)
        df = df.loc[df['workflow_version'] >= min_id]

        fn = lambda x: os.path.join(args.outpath, x)
        accumulators = [
            AccumulateFoveaMarks(fn('fovea_data.csv')), 
            AccumulateOpticNerveBox(fn('optic_nerve_box_data.csv')),
            AccumulateCupDiskBoundaryBox(fn('cup_disk_data.csv')),
            AccumulateNotchHaemorrhageMarks(fn('notch_haemorrhage_marks.csv')),
            AccumulateTasks(),
            AccumulateWorkflows(),
            ##PrintSubjectInfo(),
            ##PrintRowsForTaskKey(TASK_KEY_MARK_FOVEA),
            ##PrintRowsForTaskKey(TASK_KEY_DISK_BOUNDARY),
            ##PrintRowsForTaskKey(TASK_KEY_CUP_BOUNDARY),
            ##PrintRowsForTaskKey(TASK_KEY_MARK_NOTCH_HAEMORRHAGE),
        ]
        for a in accumulators:
            a.setup(df)

        for idx, row in df.iterrows():
            rkey = row_ukey(row)
            skey = row_skey(row)
            if args.verbose: print(' Processing row: %s|%s'%(str(rkey), str(skey)))
            
            #print(yaml.dump(parse_field(row['annotations'])))
            for a in accumulators:
                a.handle_row(rkey, skey, row)
                
        for a in accumulators:
            a.finish(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to read Zooniverse input and convert to objects')    
    parser.add_argument('file', nargs='+', help='files to process')
    parser.add_argument('-o', '--outpath', help='output path', default='')
    parser.add_argument('-v', '--verbose', help='verbose output', default=False, action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print(args)

    main(args)
    
