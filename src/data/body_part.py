"""Derive an approximate examined body part from the Croatian diagnosis text.

This is NOT the DICOM BodyPartExamined tag. Our export never carried that field,
so this guesses the region from anatomical words in the report. I keep it separate
from the real metadata for that reason, and anything built on it has to say it is
derived.

Croatian declines heavily, so everything below matches on stems rather than whole
words. "mozg" catches mozak, mozga, mozgu.

Scoring: I count stem hits per category and take the category with the most hits.
Ties go to whichever category is listed first, so the specific joints sit above the
broad regions and a knee report does not get swallowed by a generic skeletal term.
"""

import re

# order matters, earlier entries win ties, so everything specific sits above the
# broad regions. the peripheral skeleton is split finely because those reports are
# about one bone or one joint. the trunk viscera deliberately are not: an abdominal
# CT names liver, spleen, pancreas and kidneys in the same paragraph, so splitting
# them would just hand the exam to whichever organ got mentioned most often
BODY_PART_STEMS = [
    # upper limb
    ('FINGER',       ['prst', 'falang']),
    ('WRIST',        ['zapesc', 'karpal', 'rucnog zgloba', 'rucni zglob', 'skafoid']),
    ('HAND',         ['sake', 'saka', 'saku', 'metakarp', 'dlan']),
    ('FOREARM',      ['podlakt', 'radiusa', 'ulne', 'ulna']),
    ('ELBOW',        ['lakat', 'lakt', 'olekran']),
    ('ARM',          ['nadlakt', 'humerus']),
    ('SHOULDER',     ['ramen', 'akromi', 'rotatorn', 'supraspinat', 'glenoid']),
    ('CLAVICLE',     ['klavikul', 'kljucn kost', 'kljucne kosti']),
    ('SCAPULA',      ['lopatic', 'skapul']),
    # lower limb
    ('HEEL',         ['petn kost', 'petne kosti', 'kalkane']),
    ('FOOT',         ['stopal', 'metatarz']),
    ('ANKLE',        ['gleznj', 'skocn', 'maleol']),
    ('LEG',          ['potkoljenic', 'tibij', 'fibul']),
    ('KNEE',         ['koljen', 'patel', 'menisk', 'krizn svez']),
    ('THIGH',        ['bedren', 'femur']),
    ('HIP',          ['kuka', 'kuku', 'kukov', 'acetabul']),
    # spine, split by level before the generic bucket
    ('CSPINE',       ['vratne kralj', 'vratna kralj', 'cervikaln kralj', 'c-kralj']),
    ('TSPINE',       ['torakaln kralj', 'grudne kralj', 'grudn kralj']),
    ('LSPINE',       ['lumbaln', 'lumbosakral', 'ls kralj', 'l-s kralj']),
    ('SACRUM',       ['sakrum', 'sakraln', 'trticn', 'kokcig']),
    ('SPINE',        ['kraljesnic', 'vertebr', 'intervertebr', 'diskus', 'diskopat']),
    ('RIBS',         ['rebr']),
    ('STERNUM',      ['sternum', 'prsne kosti']),
    # head and neck
    ('ORBIT',        ['orbit', 'ocne supljin', 'bulbus']),
    ('SINUS',        ['paranazaln', 'sinusit', 'maksilarn sinus', 'etmoid',
                      'frontaln sinus', 'sfenoidn sinus']),
    ('MANDIBLE',     ['mandibul', 'donje celjust', 'celjusn zglob',
                      'temporomandibul']),
    ('SKULL',        ['lubanj', 'lubanje', 'kranij', 'kalvarij', 'piramid']),
    ('HEAD',         ['mozg', 'cerebr', 'cerebel', 'intrakranij', 'hipofiz',
                      'ventrikul', 'temporaln reznj', 'frontaln reznj']),
    ('THYROID',      ['stitnjac', 'tiroid']),
    ('CAROTID',      ['karotid']),
    ('NECK',         ['vratnih', 'submandibular', 'vratne regije']),
    ('BREAST',       ['dojk', 'mamograf']),
    # trunk, kept as regions on purpose, see the note above
    ('URINARYTRACT', ['bubre', 'renaln', 'mokrac', 'mjehur', 'ureter', 'uretr',
                      'nadbubrez']),
    ('GITRACT',      ['crijev', 'kolon', 'zeludac', 'zeluc', 'duoden', 'ezofag',
                      'jednjak', 'rektum', 'sigm']),
    ('PELVIS',       ['zdjelic', 'prostat', 'maternic', 'jajnik', 'sakroilij']),
    ('ABDOMEN',      ['abdomen', 'trbuh', 'jetr', 'hepat', 'slezen', 'pankreas',
                      'gusterac', 'zucn', 'peritone']),
    ('CHEST',        ['toraks', 'torakaln', 'pluc', 'grudn', 'pleur', 'medijastin',
                      'srca', 'srce', 'perikard', 'bronh']),
    ('VASCULAR',     ['arterij', 'aort', 'angiograf', 'embolizac', 'vensk',
                      'tromboz']),
]


def _normalise(text):
    # strip diacritics so a stem written either way still matches. cheap and good
    # enough here, I am matching stems not doing real linguistics
    t = str(text).lower()
    for a, b in (('č', 'c'), ('ć', 'c'), ('ž', 'z'), ('š', 's'), ('đ', 'd')):
        t = t.replace(a, b)
    return t


def _pattern(stem):
    """Turn a stem into a regex that tolerates Croatian case endings.

    A space in a stem becomes "any ending, then whitespace", so "torakaln kralj"
    still matches "torakalne kraljeznice". Without this every multi word stem
    silently matches nothing, which is how TSPINE first came out at zero.
    """
    return r'\w*\s+'.join(re.escape(p) for p in _normalise(stem).split())


_COMPILED = [(label, [re.compile(_pattern(s)) for s in stems])
             for label, stems in BODY_PART_STEMS]


def derive_body_part(text, unknown='UNCLASSIFIED'):
    """Best guess at the examined region, or `unknown` when nothing matches."""
    t = _normalise(text)
    best, best_hits = unknown, 0
    for label, pats in _COMPILED:
        hits = sum(len(p.findall(t)) for p in pats)
        if hits > best_hits:
            best, best_hits = label, hits
    return best


def add_body_part(df, text_col='DIAGNOSIS_HR', out_col='BODY_PART_DERIVED'):
    """Attach the derived column to a frame that carries the diagnosis text."""
    df = df.copy()
    df[out_col] = df[text_col].map(derive_body_part)
    return df
