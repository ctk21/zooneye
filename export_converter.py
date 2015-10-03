
import argparse
import math
import numpy as np
import os
import pandas as pd
import sys
import yaml

## valid workflow id and minimum versions supported
VALID_WORKFLOWS = {
    162: 16.67,        ## 'NerveMarking1'
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
    ## handle multiple versions
    fnme_key = 'Filename' if 'Filename' in sub_dat[sub_key] else 'filename' 
    return {'subject_id': sub_key, 'subject_filename': sub_dat[sub_key][fnme_key]}

def parse_field(field):
    return yaml.safe_load(field) if type(field) is str else field

def parse_point_array(point_list):
    return pd.DataFrame(point_list).values

def is_task_key(task_key, workflow_id, annotation_row, skip_empty_value=True):
    if workflow_id != task_key[0]:
        return False
    if ('task' in annotation_row) and ('task_label' in annotation_row):        
        if annotation_row['task'] == task_key[1]:
            if not skip_empty_value:
                return True
            else:
                arv = annotation_row['value']
                return (not arv is None) and (len(annotation_row['value']) > 0)
        return False        
    else:
        print('No task, task_label with value in annotation: %s'%str(annotation_row))
        return False

def push_keys_and_dict_onto_list(ukey, skey, rdict, the_list):
    rdict.update(ukey)
    rdict.update(skey)
    the_list.append(rdict)
    return the_list

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def nan2value(x, v):
    return x if not math.isnan(x) else v

def calc_f2d_mu_scale(df):
    return 4500./np.sqrt((df['fovea_center_x'].values - df['disk_center_x'].values)**2 + (df['fovea_center_y'].values - df['disk_center_y'].values)**2)

class BaseAccumulator(object):
    def setup(self, df): pass        
    def handle_row(self, rkey, skey, row): pass
    def finish(self, df): pass

class DataFrameAccumulator(BaseAccumulator):
    def __init__(self):
        self.row_list = []
        self.stat_df = None

    def dataframe(self):
        if self.stat_df is None:
            self.stat_df = df_xcols(pd.DataFrame(self.row_list), CSV_KEY_ORDER)
        return self.stat_df
    
    
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


class AccumulateOpticNerveBox(DataFrameAccumulator):
    def __init__(self, out_file=None):
        super(AccumulateOpticNerveBox, self).__init__()
        self.out_file = out_file
        
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_DISK_BOX, workflow_id, x):
                dat = x['value']
                ## handle multiple versions
                if type(dat) is list:
                    dat = dat[0]
                rdict = {'nerve_box_height': dat['height'],
                         'nerve_box_width': dat['width'],
                         'nerve_box_center_x': dat['x']+0.5*dat['width'],
                         'nerve_box_center_y': dat['y']+0.5*dat['height'],}
                push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)
                
    def finish(self, df):        
        stat_df = self.dataframe()
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['nerve_box_center_x'].agg({'n': len}),
                       grouped[['nerve_box_center_x', 'nerve_box_center_y', 'nerve_box_width', 'nerve_box_height']].agg(np.mean))
        print('Optic Nerve Box statistics: ')
        print(tmp)
        print('Count Optic Nerve Box rows: %i'%len(stat_df))
        print('Mean Optic Nerve Box height/width: %.4f'%np.mean(stat_df['nerve_box_height'].values/stat_df['nerve_box_width'].values))

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
        
            
class AccumulateFoveaMarks(DataFrameAccumulator):
    def __init__(self, out_file=None):
        super(AccumulateFoveaMarks, self).__init__()
        self.out_file = out_file
        
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_MARK_FOVEA, workflow_id, x):
                dat = x['value'][0]
                rdict = {'fovea_center_x': dat['x'],
                         'fovea_center_y': dat['y'],}
                push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)
                
    def finish(self, df):        
        stat_df = self.dataframe()
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['fovea_center_x'].agg({'n': len}),
                       grouped[['fovea_center_x', 'fovea_center_y']].agg(np.mean))
        print('Fovea mark statistics: ')
        print(tmp)

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
                

class AccumulateCupDiskBoundaryBox(DataFrameAccumulator):
    def __init__(self, out_file=None):
        super(AccumulateCupDiskBoundaryBox, self).__init__()
        self.out_file = out_file
        
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        valid_disk = valid_cup = False
        for x in annotations:
            if is_task_key(TASK_KEY_DISK_BOUNDARY, workflow_id, x):
                dat = x['value']
                if not 'points' in dat[-1]:
                    print('WARNING: skipping disk boundary as no points field: %s'%str(x))
                else:
                    dat = dat[-1]
                    assert(dat['tool'] == 0)
                    points = parse_point_array(dat['points'])

                    disk_area = poly_area(points[:,1], points[:,0])
                    disk_min = np.min(points, axis=0)
                    disk_max = np.max(points, axis=0)                    
                    valid_disk = True

            if is_task_key(TASK_KEY_CUP_BOUNDARY, workflow_id, x):
                dat = x['value']
                if not 'points' in dat[-1]:
                    print('WARNING: skipping cup boundary as no points field: %s'%str(x))
                else:                    
                    dat = dat[-1]
                    assert(dat['tool'] == 0)
                    points = parse_point_array(dat['points'])

                    cup_area = poly_area(points[:,1], points[:,0])
                    cup_min = np.min(points, axis=0)
                    cup_max = np.max(points, axis=0)
                    valid_cup = True
        
        if valid_cup and valid_disk:
            rdict = {
                'disk_center_x': 0.5*(disk_min[0]+disk_max[0]),
                'disk_center_y': 0.5*(disk_min[1]+disk_max[1]),
                'disk_width': disk_max[0] - disk_min[0],
                'disk_height': disk_max[1] - disk_min[1],
                'disk_area': disk_area,
                'cup_center_x': 0.5*(cup_min[0]+cup_max[0]),
                'cup_center_y': 0.5*(cup_min[1]+cup_max[1]),
                'cup_width': cup_max[0] - cup_min[0],
                'cup_height': cup_max[1] - cup_min[1],
                'cup_area': cup_area,
            }
            rdict['cdr_vertical'] = rdict['cup_height']/rdict['disk_height']
            rdict['cdr_horizontal'] = rdict['cup_width']/rdict['disk_width']
            rdict['cdr_area'] = math.sqrt(rdict['cup_area']/rdict['disk_area'])
            rdict['nerve_cd_area'] = rdict['disk_area'] - rdict['cup_area']
            push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)
            
    def finish(self, df):
        stat_df = self.dataframe()
        grouped = stat_df.groupby(['subject_id'], as_index=False)
        tmp = pd.merge(grouped['disk_center_x'].agg({'n': len}),
                       grouped[['cdr_vertical', 'cdr_horizontal', 'cdr_area', 'nerve_cd_area']].agg(np.mean))

        print('CupDiskBoundaryBox statistics: ')
        print(tmp)

        if self.out_file:
            stat_df.to_csv(self.out_file, index=False)
        
   
class AccumulateNotchHaemorrhageMarks(DataFrameAccumulator):    
    def __init__(self, out_file=None):
        super(AccumulateNotchHaemorrhageMarks, self).__init__()
        self.out_file = out_file
    
    def handle_row(self, rkey, skey, row):
        workflow_id = row['workflow_id']
        annotations = parse_field(row['annotations'])
        for x in annotations:
            if is_task_key(TASK_KEY_MARK_NOTCH_HAEMORRHAGE, workflow_id, x, skip_empty_value=False):
                for mark in x['value']:
                    rdict = { 'mark_id': mark['tool'],
                              'mark_label': mark['tool_label'],
                              'mark_center_x': mark['x'],
                              'mark_center_y': mark['y'], }
                    push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)    

                if len(x['value']) == 0:
                    rdict = { 'mark_id': -1, 
                              'mark_label': 'No_Notch_Or_Haemorrhage',
                              'mark_center_x': -1,
                              'mark_center_y': -1, }
                    push_keys_and_dict_onto_list(rkey, skey, rdict, self.row_list)    

    def dataframe(self):
        NOTCH_ID = 0
        HAEMORRHAGE_ID = 1
        df = super(AccumulateNotchHaemorrhageMarks, self).dataframe()
        grouped = df.groupby(CSV_KEY_ORDER, as_index=False)
        return grouped['mark_id'].agg({
            'n_notch': lambda xs: np.sum(xs == NOTCH_ID),
            'n_heamorrhage': lambda xs: np.sum(xs == HAEMORRHAGE_ID),
        }) 
        
    def finish(self, df):        
        stat_df = super(AccumulateNotchHaemorrhageMarks, self).dataframe()
        grouped = stat_df.groupby(['subject_id', 'mark_id', 'mark_label'], as_index=False)
        tmp = grouped['mark_center_x'].agg({'n': len})
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

                
def ParseExpertCSV(args):
    expert_set = set()
    if args.expert_csv is not None:
        if args.verbose: print('Reading expert set from: '+args.expert_csv)
        df = pd.read_csv(args.expert_csv)
        expert_set = set(df['user_name'].values)
    return expert_set

        
def main(args):
    expert_set = ParseExpertCSV(args)
    
    for fnme in args.file:
        if args.verbose: print('Processing: '+fnme)
        
        df = pd.read_csv(fnme)   
        if args.verbose: df.info()

        ## filter to valid workflows
        df = df.loc[df['workflow_id'].isin(VALID_WORKFLOWS.keys())]
        min_id = df['workflow_id'].replace(VALID_WORKFLOWS)
        df = df.loc[df['workflow_version'] >= min_id]

        outdir_fn = lambda x: os.path.join(args.outpath, x)
        accum_fovea = AccumulateFoveaMarks(outdir_fn('fovea_data.csv'))
        accum_optic_box = AccumulateOpticNerveBox(outdir_fn('optic_nerve_box_data.csv'))
        accum_cup_disk = AccumulateCupDiskBoundaryBox(outdir_fn('cup_disk_data.csv'))
        accum_notch_haem = AccumulateNotchHaemorrhageMarks(outdir_fn('notch_haemorrhage_marks.csv'))
        df_accumulators = [
            accum_fovea,
            accum_optic_box,
            accum_cup_disk,
            accum_notch_haem,
        ]
        
        accumulators = df_accumulators + [
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
            if args.dump_annotations: print(yaml.dump(parse_field(row['annotations'])))
            if rkey['user_name'] in expert_set: rkey['expert'] = 1
            for a in accumulators:
                a.handle_row(rkey, skey, row)
                
        for a in accumulators:
            a.finish(df)

        ## create single dataframe and calculate mu_scale
        merged_df = df_accumulators[0].dataframe()
        for dfa in df_accumulators[1:]:
            merged_df = pd.merge(merged_df, dfa.dataframe(), how='outer')
        merged_df['f2d_mu_scale'] = calc_f2d_mu_scale(merged_df)
        merged_df['nerve_cd_area_mu2'] = merged_df['nerve_cd_area']*(merged_df['f2d_mu_scale']**2)
        merged_df = df_xcols(merged_df, CSV_KEY_ORDER)
        merged_df.to_csv(outdir_fn('all_data.csv'), index=False)

        ## create aggregated cdr ratios by subject
        def group_and_describe(df, group_key, trgt_nme):
            tmp = df.groupby(group_key)
            if len(tmp) == 0: return pd.DataFrame()
            tmp = tmp[trgt_nme].describe().unstack()
            tmp.columns = [trgt_nme+'_'+n for n in tmp.columns.values]
            return tmp
        
        def fn(merged_df, trgt_nme):
            sub_key = ['subject_id', 'subject_filename']
            tmp_df = merged_df[sub_key+['expert', trgt_nme]]
            ok = np.isfinite(tmp_df[trgt_nme].values)
            
            normal_df = group_and_describe(tmp_df[(tmp_df.expert==0) & ok], sub_key, trgt_nme) 
            expert_df = group_and_describe(tmp_df[(tmp_df.expert>0) & ok], sub_key, trgt_nme)
            if len(expert_df) == 0: return normal_df
            return normal_df.join(expert_df, how='outer', rsuffix='_expert')

        tmp = fn(merged_df, 'cdr_horizontal')
        tmp.to_csv(outdir_fn('cdr_horizontal_aggregate.csv'))
        tmp = fn(merged_df, 'cdr_vertical')
        tmp.to_csv(outdir_fn('cdr_vertical_aggregate.csv'))

        print('Total classifications by normal/expert:')
        print(merged_df.groupby('expert')['expert'].count())
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tool to read Zooniverse input and convert to objects')    
    parser.add_argument('file', nargs='+', help='files to process')
    parser.add_argument('-o', '--outpath', help='output path', default='')
    parser.add_argument('-e', '--expert_csv', help='csv (name, user_name) listing expert users', default=None)
    parser.add_argument('-v', '--verbose', help='verbose output', default=False, action='store_true')
    parser.add_argument('--dump_annotations', help='dump every parsed annotation field', default=False, action='store_true')
    
    args = parser.parse_args()

    if args.verbose:
        print(args)

    main(args)
    
