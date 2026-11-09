import struct,sys
from pathlib import Path
def var(b,i):
 v=0;shift=0
 while 1:
  x=b[i];i+=1;v|=(x&127)<<shift
  if x<128:return v,i
  shift+=7
def parse(b):
 i=0;fields=[]
 while i<len(b):
  x,i=var(b,i);n=x>>3;w=x&7
  if not n:raise ValueError('field0')
  if w==0:v,i=var(b,i)
  elif w==1:v=b[i:i+8];i+=8
  elif w==2:
   l,i=var(b,i);v=b[i:i+l];i+=l
  elif w==5:v=b[i:i+4];i+=4
  else:raise ValueError(w)
  fields.append((n,w,v))
 return fields
def disp(b,depth=0):
 for n,w,v in parse(b):
  if w==1: print(' '*depth,n,w,'double',struct.unpack('<d',v)[0])
  elif w==5: print(' '*depth,n,w,'float',struct.unpack('<f',v)[0])
  elif w==0: print(' '*depth,n,w,v)
  else:
   print(' '*depth,n,w,'len',len(v),'hex',v[:40].hex())
   if depth<4 and len(v)<300:
    try:disp(v,depth+1)
    except:pass
if __name__=='__main__':disp(Path(sys.argv[1]).read_bytes())
