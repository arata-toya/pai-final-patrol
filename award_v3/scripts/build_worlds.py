#!/usr/bin/env python3
from pathlib import Path
import json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
SCENES={
 'CASE_A':[('normal',4),('stain',7),('crack',10)],
 'CASE_B':[('normal',4),('crack_alt',7),('stain',10)],
 'CASE_C':[('normal',4),('stain',7),('normal',10)],
 'CASE_D':[('normal',4),('crack',7),('crack_alt',10)]}

def build():
    manifest={}
    for case,panels in SCENES.items():
        tree=ET.parse(ROOT/'worlds/base.sdf');sdf=tree.getroot();sdf.set('version','1.8')
        world=sdf.find('world');world.set('name','wall_v3_'+case.lower())
        scene=world.find('scene')
        if scene is None:scene=ET.SubElement(world,'scene')
        ET.SubElement(scene,'ambient').text='.8 .8 .8 1'
        ET.SubElement(scene,'background').text='.75 .78 .80 1'
        for m in list(world.findall('model')):
            if m.get('name') not in ('ground_plane','vehicle_blue'):world.remove(m)
        sensor=world.find("model[@name='vehicle_blue']/link[@name='chassis']/sensor")
        sensor.find('pose').text='1.3 0 0.45 0 0 0.45'
        sensor.find('topic').text='/wall/camera'
        sensor.find('camera/horizontal_fov').text='1.2'
        sensor.find('camera/image/width').text='640';sensor.find('camera/image/height').text='480'
        sensor.find('update_rate').text='8'
        # 巡視路・連続壁・水面。画像のクラスを示す色は使用しない。
        for name,pose,size,color in [
            ('maintenance_path','6 1.3 .007 0 0 0','20 2.8 .01','.40 .42 .43 1'),
            ('continuous_concrete_wall','6 3.62 1.15 0 0 0','20 .2 1.7','.60 .61 .60 1'),
            ('river_water','6 7.5 .01 0 0 0','40 6 .015','.16 .43 .56 1'),
            ('grass_verge','6 -1.3 .006 0 0 0','40 2 .01','.32 .46 .26 1')]:
            m=ET.SubElement(world,'model',name=name);ET.SubElement(m,'static').text='true'
            ET.SubElement(m,'pose').text=pose;ln=ET.SubElement(m,'link',name='link')
            v=ET.SubElement(ln,'visual',name='visual');g=ET.SubElement(v,'geometry')
            ET.SubElement(ET.SubElement(g,'box'),'size').text=size
            mt=ET.SubElement(v,'material');ET.SubElement(mt,'ambient').text=color;ET.SubElement(mt,'diffuse').text=color
            if name=='continuous_concrete_wall':
                co=ET.SubElement(ln,'collision',name='collision');cg=ET.SubElement(co,'geometry')
                ET.SubElement(ET.SubElement(cg,'box'),'size').text=size
        for i,(label,x) in enumerate(panels):
            m=ET.SubElement(world,'model',name=f'wall_section_{i+1}')
            ET.SubElement(m,'static').text='true'
            ET.SubElement(m,'pose').text=f'{x} 3.45 1.2 0 0 0'
            link=ET.SubElement(m,'link',name='panel')
            visual=ET.SubElement(link,'visual',name='concrete_surface')
            geo=ET.SubElement(visual,'geometry');plane=ET.SubElement(geo,'plane')
            ET.SubElement(plane,'normal').text='0 -1 0';ET.SubElement(plane,'size').text='1.6 1.6'
            mat=ET.SubElement(visual,'material')
            ET.SubElement(mat,'ambient').text='1 1 1 1'
            ET.SubElement(mat,'diffuse').text='1 1 1 1'
            ET.SubElement(mat,'specular').text='0 0 0 1'
            metal=ET.SubElement(ET.SubElement(mat,'pbr'),'metal')
            ET.SubElement(metal,'albedo_map').text=(ROOT/'assets/textures'/f'{label}.png').as_uri()
            ET.SubElement(metal,'roughness').text='1';ET.SubElement(metal,'metalness').text='0'
            # 青緑枠と画像は全クラス共通の位置決め用。衝突形状は道路脇に置く。
            for tag in ('visual','collision'):
                b=ET.SubElement(link,tag,name='backing');ET.SubElement(b,'pose').text='0 .05 0 0 0 0'
                box=ET.SubElement(ET.SubElement(b,'geometry'),'box');ET.SubElement(box,'size').text='1.6 .08 1.6'
                if tag=='visual':
                    mt=ET.SubElement(b,'material');ET.SubElement(mt,'ambient').text='.55 .55 .55 1';ET.SubElement(mt,'diffuse').text='.55 .55 .55 1'
        ET.indent(tree,space='  ');tree.write(ROOT/'worlds'/f'{case}.sdf',encoding='utf-8',xml_declaration=True)
        manifest[case]=dict(panels=[dict(class_name='crack' if label=='crack_alt' else label,
                                       texture=label,x_m=x) for label,x in panels],end_x_m=12.3)
    (ROOT/'scenarios.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':build()
