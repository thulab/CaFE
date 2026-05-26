import { createI18n } from 'vue-i18n';
import enUS from './locales/en-US';
import zhCN from './locales/zh-CN';
import { DEFAULT_LOCALE, STORAGE_KEY, type LocaleCode, normalizeLocale, resolveInitialLocale } from './keys';

export const messages = {
  'en-US': enUS,
  'zh-CN': zhCN,
};

export const i18n = createI18n({
  legacy: false,
  locale: resolveInitialLocale(),
  fallbackLocale: DEFAULT_LOCALE,
  messages,
});

export function setDocumentLocale(locale: LocaleCode) {
  document.documentElement.lang = locale;
}

export function setLocale(locale: LocaleCode) {
  i18n.global.locale.value = locale;
  setDocumentLocale(locale);
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch (_error) {
    // Ignore storage failures; the runtime locale still changes for this session.
  }
}

export function setLocaleFromUnknown(value: unknown): LocaleCode {
  const next = normalizeLocale(value) ?? DEFAULT_LOCALE;
  setLocale(next);
  return next;
}

setDocumentLocale(i18n.global.locale.value as LocaleCode);
