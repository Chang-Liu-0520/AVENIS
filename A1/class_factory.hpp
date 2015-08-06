#include <string>
#include <vector>
#include <memory>
#include <map>

#ifndef CLASS_FACTORY
#define CLASS_FACTORY

template <typename Base>
class Base_Factory
{
 public:
  Base_Factory()
  {
  }
  virtual Base create() = 0;
};

template <typename Derived, typename Base, typename ArgType>
class Derived_Factory : public Base_Factory<std::unique_ptr<Base>>
{
 public:
  Derived_Factory(ArgType name)
  {
    Base::registerType(name, this);
  }
  std::unique_ptr<Base> create()
  {
    std::unique_ptr<Base> Derived_Instance(new Derived);
    return Derived_Instance;
  }
};

template <typename Base, typename ArgType>
class Base_Template
{
 public:
  /* Constructor */
  Base_Template()
  {
  }
  /* Virtual Destructor: Since this is the base class for lots of other
     classes. */
  virtual ~Base_Template()
  {
  }

  static void registerType(ArgType name, Base_Factory<std::unique_ptr<Base>> *factory)
  {
    get_factory_instance()[name] = factory;
  }

  static bool create(ArgType name, std::unique_ptr<Base> &TheInstance)
  {
    if (get_factory_instance().find(name) != get_factory_instance().end())
    {
      TheInstance = std::move(get_factory_instance()[name]->create());
      return true;
    }
    else
    {
      return false;
    }
  }

 protected:
  static std::map<ArgType, Base_Factory<std::unique_ptr<Base>> *> &get_factory_instance()
  {
    static std::map<ArgType, Base_Factory<std::unique_ptr<Base>> *> map_instance;
    return map_instance;
  }
};

#endif // CLASS_FACTORY
