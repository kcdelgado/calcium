#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:32 2021

@author: katherinedelgado and Erin Barnhart
"""
import numpy as numpy
from scipy import signal

class Response(object):
    """class attributes"""
    
    """instance attributes"""
    _instance_data = {'sample_name':None,
                       'ROI_num':None,
                       'reporter_name':None,
                       'driver_name':None,
                       'F':[],
                       'stimulus_name':None,
                       'stim_time':[],
                       'stim_state':[],
                       'time_step':None,
                       'units':'seconds'}
    
    def __init__(self, **kws):
        
        """self: is the instance, __init__ takes the instace 'self' and populates it with variables"""
        
        for attr, value in self._instance_data.items():
            if attr in kws:
                value = kws[attr]
                setattr(self,attr,value)
            else:
                setattr(self,attr,value)
        
    """instance methods"""
    
    
    """find median fluorescence over time"""
    def med(self):
        return numpy.median(self.fluorescence)
        #self.median.append(median)

    """identify stim switch points"""
    def stimswitch(self):
        ON_indices = list(numpy.where(numpy.diff(self.stim_state)==1)[0]+1)
        OFF_indices = list(numpy.where(numpy.diff(self.stim_state)==-1)[0]+1)
        return ON_indices, OFF_indices
    
    """break data into epochs"""
    def segment_responses(self,points_before,points_after):
        """points_before is the number of points before ON
        points_after is the number of points after OFF"""
        ON,OFF = self.stimswitch()
        r = []
        for on, off in zip(ON,OFF):
            start = on - points_before
            stop = off + points_after
            if start>0 and stop<len(self.F)+1:
                r.append(self.F[start:stop])
                print(len(self.F[start:stop]))
        self.individual_responses = r
        return r
    
    """get df/f"""
    def dff(self,baseline_start,baseline_stop):
        #b = self.baseline_end
        ir = numpy.asarray(self.individual_responses)
        dff = []
        for i in ir:
            baseline = numpy.median(i[baseline_start:baseline_stop])
            dff.append((list(i-baseline)/baseline))
        return dff
    
    def average_dff(self,baseline_start,baseline_stop,epoch_length):
        #b = self.baseline_end
        ir = numpy.asarray(self.individual_responses)
        dff = []
        for i in ir:
            i = i[:epoch_length] #clunky solution, need to fix
            baseline = numpy.median(i[baseline_start:baseline_stop])
            dff.append((list(i-baseline)/baseline))
        D = numpy.asarray(dff)
        self.average_dff = list(numpy.average(D,axis=0))
        return numpy.average(D,axis=0)

    def stdev_dff(self,baseline_start,baseline_stop):
        #b = self.baseline_end
        ir = numpy.asarray(self.individual_responses)
        dff = []
        for i in ir:
            baseline = numpy.median(i[baseline_start:baseline_stop])
            dff.append((list(i-baseline)/baseline))
        D = numpy.asarray(dff)
        return numpy.std(D,axis=0)

    def integrated_response(self,start_point,end_point):
        IR = numpy.sum(self.average_dff[start_point:end_point])
        return IR