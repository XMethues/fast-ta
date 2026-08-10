//! Local definitions for crow, soldier, and advance patterns.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::Result;

fn maximum_average_period(settings: CandleSettings, referenced: &[CandleSettingType]) -> usize {
    referenced.iter().map(|&kind| settings.setting(kind).average_period()).max().unwrap_or(0)
}

#[inline]
const fn signal(direction: PatternDirection) -> PatternSignal {
    PatternSignal::Match { direction, strength: PatternStrength::Standard }
}

macro_rules! define_config {
    ($config:ident, $runner:ident, $stream:ident, $span:expr, [$($setting:expr),+ $(,)?]) => {
        #[doc = concat!("Immutable ", stringify!($config), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config { candle_settings: CandleSettings }
        impl $config {
            pub fn new(candle_settings: CandleSettings) -> Result<Self> { Ok(Self { candle_settings }) }
            #[inline] pub const fn candle_settings(&self) -> CandleSettings { self.candle_settings }
            #[inline] pub fn warm_up(&self) -> usize { maximum_average_period(self.candle_settings, &[$($setting),+]) + $span }
        }
        impl Default for $config { fn default() -> Self { Self { candle_settings: CandleSettings::default() } } }
        impl_pattern_execution!($config, $runner, $stream);
    };
}

macro_rules! definition {
    ($config:ident, $name:literal, [$($setting:expr),+ $(,)?], $body:expr) => {
        impl PatternDefinition for $config {
            type State = ();
            fn name(&self) -> &'static str { $name }
            fn settings(&self) -> CandleSettings { self.candle_settings }
            fn referenced_settings(&self) -> &'static [CandleSettingType] { &[$($setting),+] }
            fn lookback(&self) -> usize { self.warm_up() }
            fn transition_start(&self) -> usize { self.lookback() }
            fn initial_state(&self) -> Self::State {}
            fn transition(&self, c: &RecognitionContext<'_>, _: &mut Self::State) -> PatternSignal { $body(c) }
        }
    };
}

define_config!(CDL3BLACKCROWSConfig, CDL3BLACKCROWSBatchRunner, CDL3BLACKCROWSStream, 3, [CandleSettingType::ShadowVeryShort]);
definition!(CDL3BLACKCROWSConfig, "CDL3BLACKCROWS", [CandleSettingType::ShadowVeryShort], |c: &RecognitionContext<'_>| {
    let p=c.candle(3); let a=c.candle(2); let b=c.candle(1); let d=c.candle(0);
    if c.color(3)==CandleColor::White && c.color(2)==CandleColor::Black && c.color(1)==CandleColor::Black && c.color(0)==CandleColor::Black
        && b.open<a.open && b.open>a.close && d.open<b.open && d.open>b.close && p.high>a.close && a.close>b.close && b.close>d.close
        && c.lower_shadow(2)<c.average(CandleSettingType::ShadowVeryShort,2) && c.lower_shadow(1)<c.average(CandleSettingType::ShadowVeryShort,1)
        && c.lower_shadow(0)<c.average(CandleSettingType::ShadowVeryShort,0) { signal(PatternDirection::Bearish) } else { PatternSignal::NoMatch }
});

define_config!(CDL3STARSINSOUTHConfig, CDL3STARSINSOUTHBatchRunner, CDL3STARSINSOUTHStream, 2, [CandleSettingType::BodyLong,CandleSettingType::ShadowLong,CandleSettingType::ShadowVeryShort,CandleSettingType::BodyShort]);
definition!(CDL3STARSINSOUTHConfig, "CDL3STARSINSOUTH", [CandleSettingType::BodyLong,CandleSettingType::ShadowLong,CandleSettingType::ShadowVeryShort,CandleSettingType::BodyShort], |c: &RecognitionContext<'_>| {
    let a=c.candle(2); let b=c.candle(1); let d=c.candle(0);
    if c.color(2)==CandleColor::Black && c.color(1)==CandleColor::Black && c.color(0)==CandleColor::Black
        && c.real_body(2)>c.average(CandleSettingType::BodyLong,2) && c.lower_shadow(2)>c.average(CandleSettingType::ShadowLong,2)
        && c.real_body(1)<c.real_body(2) && b.open>a.close && b.open<=a.high && b.low<a.close && b.low>=a.low
        && c.lower_shadow(1)>c.average(CandleSettingType::ShadowVeryShort,1) && c.real_body(0)<c.average(CandleSettingType::BodyShort,0)
        && c.lower_shadow(0)<c.average(CandleSettingType::ShadowVeryShort,0) && c.upper_shadow(0)<c.average(CandleSettingType::ShadowVeryShort,0)
        && d.low>b.low && d.high<b.high { signal(PatternDirection::Bullish) } else { PatternSignal::NoMatch }
});

define_config!(CDL3WHITESOLDIERSConfig, CDL3WHITESOLDIERSBatchRunner, CDL3WHITESOLDIERSStream, 2, [CandleSettingType::ShadowVeryShort,CandleSettingType::BodyShort,CandleSettingType::Far,CandleSettingType::Near]);
definition!(CDL3WHITESOLDIERSConfig, "CDL3WHITESOLDIERS", [CandleSettingType::ShadowVeryShort,CandleSettingType::BodyShort,CandleSettingType::Far,CandleSettingType::Near], |c: &RecognitionContext<'_>| {
    let a=c.candle(2); let b=c.candle(1); let d=c.candle(0);
    if c.color(2)==CandleColor::White && c.color(1)==CandleColor::White && c.color(0)==CandleColor::White
        && c.upper_shadow(2)<c.average(CandleSettingType::ShadowVeryShort,2) && c.upper_shadow(1)<c.average(CandleSettingType::ShadowVeryShort,1) && c.upper_shadow(0)<c.average(CandleSettingType::ShadowVeryShort,0)
        && d.close>b.close && b.close>a.close && b.open>a.open && b.open<=a.close+c.average(CandleSettingType::Near,2)
        && d.open>b.open && d.open<=b.close+c.average(CandleSettingType::Near,1)
        && c.real_body(1)>c.real_body(2)-c.average(CandleSettingType::Far,2) && c.real_body(0)>c.real_body(1)-c.average(CandleSettingType::Far,1)
        && c.real_body(0)>c.average(CandleSettingType::BodyShort,0) { signal(PatternDirection::Bullish) } else { PatternSignal::NoMatch }
});

define_config!(CDLADVANCEBLOCKConfig, CDLADVANCEBLOCKBatchRunner, CDLADVANCEBLOCKStream, 2, [CandleSettingType::ShadowLong,CandleSettingType::ShadowShort,CandleSettingType::Far,CandleSettingType::Near,CandleSettingType::BodyLong]);
definition!(CDLADVANCEBLOCKConfig, "CDLADVANCEBLOCK", [CandleSettingType::ShadowLong,CandleSettingType::ShadowShort,CandleSettingType::Far,CandleSettingType::Near,CandleSettingType::BodyLong], |c: &RecognitionContext<'_>| {
    let a=c.candle(2); let b=c.candle(1); let d=c.candle(0); let ab=c.real_body(2); let bb=c.real_body(1); let db=c.real_body(0);
    let weakening=(bb<ab-c.average(CandleSettingType::Far,2) && db<bb+c.average(CandleSettingType::Near,1))
        || db<bb-c.average(CandleSettingType::Far,1)
        || (db<bb && bb<ab && (c.upper_shadow(0)>c.average(CandleSettingType::ShadowShort,0) || c.upper_shadow(1)>c.average(CandleSettingType::ShadowShort,1)))
        || (db<bb && c.upper_shadow(0)>c.average(CandleSettingType::ShadowLong,0));
    if c.color(2)==CandleColor::White && c.color(1)==CandleColor::White && c.color(0)==CandleColor::White && d.close>b.close && b.close>a.close
        && b.open>a.open && b.open<=a.close+c.average(CandleSettingType::Near,2) && d.open>b.open && d.open<=b.close+c.average(CandleSettingType::Near,1)
        && ab>c.average(CandleSettingType::BodyLong,2) && c.upper_shadow(2)<c.average(CandleSettingType::ShadowShort,2) && weakening
        { signal(PatternDirection::Bearish) } else { PatternSignal::NoMatch }
});

define_config!(CDLCONCEALBABYSWALLConfig, CDLCONCEALBABYSWALLBatchRunner, CDLCONCEALBABYSWALLStream, 3, [CandleSettingType::ShadowVeryShort]);
definition!(CDLCONCEALBABYSWALLConfig, "CDLCONCEALBABYSWALL", [CandleSettingType::ShadowVeryShort], |c: &RecognitionContext<'_>| {
    let b=c.candle(2); let d=c.candle(1); let e=c.candle(0);
    if c.color(3)==CandleColor::Black && c.color(2)==CandleColor::Black && c.color(1)==CandleColor::Black && c.color(0)==CandleColor::Black
        && c.lower_shadow(3)<c.average(CandleSettingType::ShadowVeryShort,3) && c.upper_shadow(3)<c.average(CandleSettingType::ShadowVeryShort,3)
        && c.lower_shadow(2)<c.average(CandleSettingType::ShadowVeryShort,2) && c.upper_shadow(2)<c.average(CandleSettingType::ShadowVeryShort,2)
        && c.real_body_gap_down(1,2) && c.upper_shadow(1)>c.average(CandleSettingType::ShadowVeryShort,1) && d.high>b.close && e.high>d.high && e.low<d.low
        { signal(PatternDirection::Bullish) } else { PatternSignal::NoMatch }
});

define_config!(CDLIDENTICAL3CROWSConfig, CDLIDENTICAL3CROWSBatchRunner, CDLIDENTICAL3CROWSStream, 2, [CandleSettingType::ShadowVeryShort,CandleSettingType::Equal]);
definition!(CDLIDENTICAL3CROWSConfig, "CDLIDENTICAL3CROWS", [CandleSettingType::ShadowVeryShort,CandleSettingType::Equal], |c: &RecognitionContext<'_>| {
    let a=c.candle(2); let b=c.candle(1); let d=c.candle(0); let e2=c.average(CandleSettingType::Equal,2); let e1=c.average(CandleSettingType::Equal,1);
    if c.color(2)==CandleColor::Black && c.color(1)==CandleColor::Black && c.color(0)==CandleColor::Black
        && c.lower_shadow(2)<c.average(CandleSettingType::ShadowVeryShort,2) && c.lower_shadow(1)<c.average(CandleSettingType::ShadowVeryShort,1) && c.lower_shadow(0)<c.average(CandleSettingType::ShadowVeryShort,0)
        && a.close>b.close && b.close>d.close && b.open<=a.close+e2 && b.open>=a.close-e2 && d.open<=b.close+e1 && d.open>=b.close-e1
        { signal(PatternDirection::Bearish) } else { PatternSignal::NoMatch }
});

define_config!(CDLSTALLEDPATTERNConfig, CDLSTALLEDPATTERNBatchRunner, CDLSTALLEDPATTERNStream, 2, [CandleSettingType::BodyLong,CandleSettingType::BodyShort,CandleSettingType::ShadowVeryShort,CandleSettingType::Near]);
definition!(CDLSTALLEDPATTERNConfig, "CDLSTALLEDPATTERN", [CandleSettingType::BodyLong,CandleSettingType::BodyShort,CandleSettingType::ShadowVeryShort,CandleSettingType::Near], |c: &RecognitionContext<'_>| {
    let a=c.candle(2); let b=c.candle(1); let d=c.candle(0);
    if c.color(2)==CandleColor::White && c.color(1)==CandleColor::White && c.color(0)==CandleColor::White && d.close>b.close && b.close>a.close
        && c.real_body(2)>c.average(CandleSettingType::BodyLong,2) && c.real_body(1)>c.average(CandleSettingType::BodyLong,1)
        && c.upper_shadow(1)<c.average(CandleSettingType::ShadowVeryShort,1) && b.open>a.open && b.open<=a.close+c.average(CandleSettingType::Near,2)
        && c.real_body(0)<c.average(CandleSettingType::BodyShort,0) && d.open>=b.close-c.real_body(0)-c.average(CandleSettingType::Near,1)
        { signal(PatternDirection::Bearish) } else { PatternSignal::NoMatch }
});
